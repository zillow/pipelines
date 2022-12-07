# Copyright 2020-2022 The Kubeflow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""KFP DSL v2 compiler.

This is an experimental implementation of KFP compiler that compiles KFP
pipeline into Pipeline IR:
https://docs.google.com/document/d/1PUDuSQ8vmeKSBloli53mp7GIvzekaY7sggg6ywy35Dk/
"""

import collections
import inspect
import re
import uuid
from typing import (Any, Callable, Dict, List, Mapping, Optional, Set, Tuple,
                    Union)

import kfp
from google.protobuf import json_format
from kfp.pipeline_spec import pipeline_spec_pb2
from kfp.v2 import dsl
from kfp.v2.compiler import pipeline_spec_builder as builder
from kfp.v2.components import utils as component_utils
from kfp.v2.components import component_factory
from kfp.v2.components import task_final_status
from kfp.v2.components.types import artifact_types, type_utils

_GroupOrTask = Union[tasks_group.TasksGroup, pipeline_task.PipelineTask]


class Compiler:
    """Experimental DSL compiler that targets the PipelineSpec IR.

    It compiles pipeline function into PipelineSpec json string.
    PipelineSpec is the IR protobuf message that defines a pipeline:
    https://github.com/kubeflow/pipelines/blob/237795539f7b85bac77435e2464367226ee19391/api/v2alpha1/pipeline_spec.proto#L8
    In this initial implementation, we only support components authored through
    Component yaml spec. And we don't support advanced features like conditions,
    static and dynamic loops, etc.

    Example::

        @dsl.pipeline(
          name='name',
          description='description',
        )
        def my_pipeline(a: int = 1, b: str = "default value"):
            ...

        kfp.v2.compiler.Compiler().compile(
            pipeline_func=my_pipeline,
            package_path='path/to/pipeline.json',
        )
    """

    def compile(
        self,
        pipeline_func: Callable[..., Any],
        package_path: str,
        pipeline_name: Optional[str] = None,
        pipeline_parameters: Optional[Mapping[str, Any]] = None,
        type_check: bool = True,
    ) -> None:
        """Compile the given pipeline function into pipeline job json.

        Args:
            pipeline_func: Pipeline function with @dsl.pipeline decorator.
            package_path: The output pipeline spec .json file path. For example,
                "~/pipeline_spec.json".
            pipeline_name: Optional; the name of the pipeline.
            pipeline_parameters: Optional; the mapping from parameter names to
                values.
            type_check: Optional; whether to enable the type check or not.
                Default is True.
        """
        type_check_old_value = kfp.TYPE_CHECK
        try:
            kfp.TYPE_CHECK = type_check
            pipeline_spec = self._create_pipeline_v2(
                pipeline_func=pipeline_func,
                pipeline_name=pipeline_name,
                pipeline_parameters_override=pipeline_parameters,
            )
            self._write_pipeline_spec_json(
                pipeline_spec=pipeline_spec,
                output_path=package_path,
            )
        finally:
            kfp.TYPE_CHECK = type_check_old_value

    def _create_pipeline_v2(
        self,
        pipeline_func: Callable[..., Any],
        pipeline_name: Optional[str] = None,
        pipeline_parameters_override: Optional[Mapping[str, Any]] = None,
    ) -> pipeline_spec_pb2.PipelineSpec:
        """Creates a pipeline instance and constructs the pipeline spec from
        it.

        Args:
            pipeline_func: The pipeline function with @dsl.pipeline decorator.
            pipeline_name: Optional; the name of the pipeline.
            pipeline_parameters_override: Optional; the mapping from parameter
                names to values.

        Returns:
            A PipelineSpec proto representing the compiled pipeline.
        """

        # Create the arg list with no default values and call pipeline function.
        # Assign type information to the PipelineChannel
        pipeline_meta = component_factory.extract_component_interface(
            pipeline_func)
        pipeline_name = pipeline_name or pipeline_meta.name

        pipeline_root = getattr(pipeline_func, 'pipeline_root', None)

        args_list = []
        signature = inspect.signature(pipeline_func)

        for arg_name in signature.parameters:
            arg_type = pipeline_meta.inputs[arg_name].type
            if not type_utils.is_parameter_type(arg_type):
                raise TypeError(
                    'The pipeline argument "{arg_name}" is viewed as an artifact'
                    ' due to its type "{arg_type}". And we currently do not '
                    'support passing artifacts as pipeline inputs. Consider type'
                    ' annotating the argument with a primitive type, such as '
                    '"str", "int", "float", "bool", "dict", and "list".'.format(
                        arg_name=arg_name, arg_type=arg_type))
            args_list.append(
                dsl.PipelineParameterChannel(
                    name=arg_name, channel_type=arg_type))

        with pipeline_context.Pipeline(pipeline_name) as dsl_pipeline:
            pipeline_func(*args_list)

        if not dsl_pipeline.tasks:
            raise ValueError('Task is missing from pipeline.')

        self._validate_exit_handler(dsl_pipeline)

        pipeline_inputs = pipeline_meta.inputs or {}

        # Verify that pipeline_parameters_override contains only input names
        # that match the pipeline inputs definition.
        pipeline_parameters_override = pipeline_parameters_override or {}
        for input_name in pipeline_parameters_override:
            if input_name not in pipeline_inputs:
                raise ValueError(
                    'Pipeline parameter {} does not match any known '
                    'pipeline argument.'.format(input_name))

        # Fill in the default values.
        args_list_with_defaults = [
            dsl.PipelineParameterChannel(
                name=input_name,
                channel_type=input_spec.type,
                value=pipeline_parameters_override.get(input_name) or
                input_spec.default,
            ) for input_name, input_spec in pipeline_inputs.items()
        ]

        # Making the pipeline group name unique to prevent name clashes with
        # templates
        pipeline_group = dsl_pipeline.groups[0]
        pipeline_group.name = uuid.uuid4().hex

        pipeline_spec = self._create_pipeline_spec(
            pipeline_args=args_list_with_defaults,
            pipeline=dsl_pipeline,
        )

        if pipeline_root:
            pipeline_spec.default_pipeline_root = pipeline_root

        return pipeline_spec

    def _write_pipeline_spec_json(
        self,
        pipeline_spec: pipeline_spec_pb2.PipelineSpec,
        output_path: str,
    ) -> None:
        """Writes pipeline spec into a json file.

        Args:
            pipeline_spec: IR pipeline spec.
            ouput_path: The file path to be written.

        Raises:
            ValueError: if the specified output path doesn't end with the
                acceptable extention.
        """
        json_text = json_format.MessageToJson(pipeline_spec, sort_keys=True)

        if output_path.endswith('.json'):
            with open(output_path, 'w') as json_file:
                json_file.write(json_text)
        else:
            raise ValueError(
                'The output path {} should ends with ".json".'.format(
                    output_path))

    def _validate_exit_handler(self,
                               pipeline: pipeline_context.Pipeline) -> None:
        """Makes sure there is only one global exit handler.

        This is temporary to be compatible with KFP v1.

        Raises:
            ValueError if there are more than one exit handler.
        """

        def _validate_exit_handler_helper(
            group: tasks_group.TasksGroup,
            exiting_task_names: List[str],
            handler_exists: bool,
        ) -> None:

            if isinstance(group, dsl.ExitHandler):
                if handler_exists or len(exiting_task_names) > 1:
                    raise ValueError(
                        'Only one global exit_handler is allowed and all ops need to be included.'
                    )
                handler_exists = True

            if group.tasks:
                exiting_task_names.extend([x.name for x in group.tasks])

            for group in group.groups:
                _validate_exit_handler_helper(
                    group=group,
                    exiting_task_names=exiting_task_names,
                    handler_exists=handler_exists,
                )

        _validate_exit_handler_helper(
            group=pipeline.groups[0],
            exiting_task_names=[],
            handler_exists=False,
        )

    def _create_pipeline_spec(
        self,
        pipeline_args: List[dsl.PipelineChannel],
        pipeline: pipeline_context.Pipeline,
    ) -> pipeline_spec_pb2.PipelineSpec:
        """Creates a pipeline spec object.

        Args:
            pipeline_args: The list of pipeline input parameters.
            pipeline: The instantiated pipeline object.

        Returns:
            A PipelineSpec proto representing the compiled pipeline.

        Raises:
            ValueError if the argument is of unsupported types.
        """
        self._validate_pipeline_name(pipeline.name)

        deployment_config = pipeline_spec_pb2.PipelineDeploymentConfig()
        pipeline_spec = pipeline_spec_pb2.PipelineSpec()

        pipeline_spec.pipeline_info.name = pipeline.name
        pipeline_spec.sdk_version = 'kfp-{}'.format(kfp.__version__)
        # Schema version 2.1.0 is required for kfp-pipeline-spec>0.1.13
        pipeline_spec.schema_version = '2.1.0'

        pipeline_spec.root.CopyFrom(
            builder.build_component_spec_for_group(
                pipeline_channels=pipeline_args,
                is_root_group=True,
            ))

        root_group = pipeline.groups[0]

        all_groups = self._get_all_groups(root_group)
        group_name_to_group = {group.name: group for group in all_groups}
        task_name_to_parent_groups, group_name_to_parent_groups = (
            self._get_parent_groups(root_group))
        condition_channels = self._get_condition_channels_for_tasks(root_group)
        name_to_for_loop_group = {
            group_name: group
            for group_name, group in group_name_to_group.items()
            if isinstance(group, dsl.ParallelFor)
        }
        inputs = self._get_inputs_for_all_groups(
            pipeline=pipeline,
            pipeline_args=pipeline_args,
            root_group=root_group,
            task_name_to_parent_groups=task_name_to_parent_groups,
            group_name_to_parent_groups=group_name_to_parent_groups,
            condition_channels=condition_channels,
            name_to_for_loop_group=name_to_for_loop_group,
        )
        dependencies = self._get_dependencies(
            pipeline=pipeline,
            root_group=root_group,
            task_name_to_parent_groups=task_name_to_parent_groups,
            group_name_to_parent_groups=group_name_to_parent_groups,
            group_name_to_group=group_name_to_group,
            condition_channels=condition_channels,
        )

        for group in all_groups:
            self._build_spec_by_group(
                pipeline_spec=pipeline_spec,
                deployment_config=deployment_config,
                group=group,
                inputs=inputs,
                dependencies=dependencies,
                rootgroup_name=root_group.name,
                task_name_to_parent_groups=task_name_to_parent_groups,
                group_name_to_parent_groups=group_name_to_parent_groups,
                name_to_for_loop_group=name_to_for_loop_group,
            )

        # TODO: refactor to support multiple exit handler per pipeline.
        if pipeline.groups[0].groups:
            first_group = pipeline.groups[0].groups[0]
            if isinstance(first_group, dsl.ExitHandler):
                exit_task = first_group.exit_task
                exit_task_name = component_utils.sanitize_task_name(
                    exit_task.name)
                exit_handler_group_task_name = component_utils.sanitize_task_name(
                    first_group.name)
                input_parameters_in_current_dag = [
                    input_name for input_name in
                    pipeline_spec.root.input_definitions.parameters
                ]
                exit_task_task_spec = builder.build_task_spec_for_exit_task(
                    task=exit_task,
                    dependent_task=exit_handler_group_task_name,
                    pipeline_inputs=pipeline_spec.root.input_definitions,
                )

                exit_task_component_spec = builder.build_component_spec_for_task(
                    task=exit_task)

                exit_task_container_spec = builder.build_container_spec_for_task(
                    task=exit_task)

                # Add exit task task spec
                pipeline_spec.root.dag.tasks[exit_task_name].CopyFrom(
                    exit_task_task_spec)

                # Add exit task component spec if it does not exist.
                component_name = exit_task_task_spec.component_ref.name
                if component_name not in pipeline_spec.components:
                    pipeline_spec.components[component_name].CopyFrom(
                        exit_task_component_spec)

                # Add exit task container spec if it does not exist.
                executor_label = exit_task_component_spec.executor_label
                if executor_label not in deployment_config.executors:
                    deployment_config.executors[
                        executor_label].container.CopyFrom(
                            exit_task_container_spec)
                    pipeline_spec.deployment_spec.update(
                        json_format.MessageToDict(deployment_config))

        return pipeline_spec

    def _validate_pipeline_name(self, name: str) -> None:
        """Validate pipeline name.

        A valid pipeline name should match ^[a-z0-9][a-z0-9-]{0,127}$.

        Args:
          name: The pipeline name.

        Raises:
          ValueError if the pipeline name doesn't conform to the regular expression.
        """
        pattern = re.compile(r'^[a-z0-9][a-z0-9-]{0,127}$')
        if not pattern.match(name):
            raise ValueError(
                'Invalid pipeline name: %s.\n'
                'Please specify a pipeline name that matches the regular '
                'expression "^[a-z0-9][a-z0-9-]{0,127}$" using '
                '`dsl.pipeline(name=...)` decorator.' % name)

    def _get_all_groups(
        self,
        root_group: tasks_group.TasksGroup,
    ) -> List[tasks_group.TasksGroup]:
        """Gets all groups (not including tasks) in a pipeline.

        Args:
            root_group: The root group of a pipeline.

        Returns:
            A list of all groups in topological order (parent first).
        """
        all_groups = []

        def _get_all_groups_helper(
            group: tasks_group.TasksGroup,
            all_groups: List[tasks_group.TasksGroup],
        ):
            all_groups.append(group)
            for group in group.groups:
                _get_all_groups_helper(group, all_groups)

        _get_all_groups_helper(root_group, all_groups)
        return all_groups

    def _get_parent_groups(
        self,
        root_group: tasks_group.TasksGroup,
    ) -> Tuple[Mapping[str, List[_GroupOrTask]], Mapping[str,
                                                         List[_GroupOrTask]]]:
        """Get parent groups that contain the specified tasks.

        Each pipeline has a root group. Each group has a list of tasks (leaf)
        and groups.
        This function traverse the tree and get ancestor groups for all tasks.

        Args:
            root_group: The root group of a pipeline.

        Returns:
            A tuple. The first item is a mapping of task names to parent groups,
            and second item is a mapping of group names to parent groups.
            A list of parent groups is a list of ancestor groups including the
            task/group itself. The list is sorted in a way that the farthest
            parent group is the first and task/group itself is the last.
        """

        def _get_parent_groups_helper(
            current_groups: List[tasks_group.TasksGroup],
            tasks_to_groups: Dict[str, List[_GroupOrTask]],
            groups_to_groups: Dict[str, List[_GroupOrTask]],
        ) -> None:
            root_group = current_groups[-1]
            for group in root_group.groups:

                groups_to_groups[group.name] = [x.name for x in current_groups
                                               ] + [group.name]
                current_groups.append(group)

                _get_parent_groups_helper(
                    current_groups=current_groups,
                    tasks_to_groups=tasks_to_groups,
                    groups_to_groups=groups_to_groups,
                )
                del current_groups[-1]

            for task in root_group.tasks:
                tasks_to_groups[task.name] = [x.name for x in current_groups
                                             ] + [task.name]

        tasks_to_groups = {}
        groups_to_groups = {}
        current_groups = [root_group]

        _get_parent_groups_helper(
            current_groups=current_groups,
            tasks_to_groups=tasks_to_groups,
            groups_to_groups=groups_to_groups,
        )
        return (tasks_to_groups, groups_to_groups)

    # TODO: do we really need this?
    def _get_condition_channels_for_tasks(
        self,
        root_group: tasks_group.TasksGroup,
    ) -> Mapping[str, Set[dsl.PipelineChannel]]:
        """Gets channels referenced in conditions of tasks' parents.

        Args:
            root_group: The root group of a pipeline.

        Returns:
            A mapping of task name to a set of pipeline channels appeared in its
            parent dsl.Condition groups.
        """
        conditions = collections.defaultdict(set)

        def _get_condition_channels_for_tasks_helper(
            group,
            current_conditions_channels,
        ):
            new_current_conditions_channels = current_conditions_channels
            if isinstance(group, dsl.Condition):
                new_current_conditions_channels = list(
                    current_conditions_channels)
                if isinstance(group.condition.left_operand,
                              dsl.PipelineChannel):
                    new_current_conditions_channels.append(
                        group.condition.left_operand)
                if isinstance(group.condition.right_operand,
                              dsl.PipelineChannel):
                    new_current_conditions_channels.append(
                        group.condition.right_operand)
            for task in group.tasks:
                for channel in new_current_conditions_channels:
                    conditions[task.name].add(channel)
            for group in group.groups:
                _get_condition_channels_for_tasks_helper(
                    group, new_current_conditions_channels)

        _get_condition_channels_for_tasks_helper(root_group, [])
        return conditions

    def _get_inputs_for_all_groups(
        self,
        pipeline: pipeline_context.Pipeline,
        pipeline_args: List[dsl.PipelineChannel],
        root_group: tasks_group.TasksGroup,
        task_name_to_parent_groups: Mapping[str, List[_GroupOrTask]],
        group_name_to_parent_groups: Mapping[str, List[tasks_group.TasksGroup]],
        condition_channels: Mapping[str, Set[dsl.PipelineParameterChannel]],
        name_to_for_loop_group: Mapping[str, dsl.ParallelFor],
    ) -> Mapping[str, List[Tuple[dsl.PipelineChannel, str]]]:
        """Get inputs and outputs of each group and op.

        Args:
            pipeline: The instantiated pipeline object.
            pipeline_args: The list of pipeline function arguments as
                PipelineChannel.
            root_group: The root group of the pipeline.
            task_name_to_parent_groups: The dict of task name to list of parent
                groups.
            group_name_to_parent_groups: The dict of group name to list of
                parent groups.
            condition_channels: The dict of task name to a set of pipeline
                channels referenced by its parent condition groups.
            name_to_for_loop_group: The dict of for loop group name to loop
                group.

        Returns:
            A mapping  with key being the group/task names and values being list
            of tuples (channel, producing_task_name).
            producing_task_name is the name of the task that produces the
            channel. If the channel is a pipeline argument (no producer task),
            then producing_task_name is None.
        """
        inputs = collections.defaultdict(set)

        for task in pipeline.tasks.values():
            # task's inputs and all channels used in conditions for that task are
            # considered.
            task_inputs = task.channel_inputs
            task_condition_inputs = list(condition_channels[task.name])

            for channel in task.channel_inputs + task_condition_inputs:

                # If the value is already provided (immediate value), then no
                # need to expose it as input for its parent groups.
                if getattr(channel, 'value', None):
                    continue

                # channels_to_add could be a list of PipelineChannels when loop
                # args are involved. Given a nested loops example as follows:
                #
                #  def my_pipeline(loop_parameter: list):
                #       with dsl.ParallelFor(loop_parameter) as item:
                #           with dsl.ParallelFor(item.p_a) as item_p_a:
                #               print_op(item_p_a.q_a)
                #
                # The print_op takes an input of
                # {{channel:task=;name=loop_parameter-loop-item-subvar-p_a-loop-item-subvar-q_a;}}.
                # Given this, we calculate the list of PipelineChannels potentially
                # needed by across DAG levels as follows:
                #
                # [{{channel:task=;name=loop_parameter-loop-item-subvar-p_a-loop-item-subvar-q_a}},
                #  {{channel:task=;name=loop_parameter-loop-item-subvar-p_a-loop-item}},
                #  {{channel:task=;name=loop_parameter-loop-item-subvar-p_a}},
                #  {{channel:task=;name=loop_parameter-loop-item}},
                #  {{chaenel:task=;name=loop_parameter}}]
                #
                # For the above example, the first loop needs the input of
                # {{channel:task=;name=loop_parameter}},
                # the second loop needs the input of
                # {{channel:task=;name=loop_parameter-loop-item}}
                # and the print_op needs the input of
                # {{channel:task=;name=loop_parameter-loop-item-subvar-p_a-loop-item}}
                #
                # When we traverse a DAG in a top-down direction, we add channels
                # from the end, and pop it out when it's no longer needed by the
                # sub-DAG.
                # When we traverse a DAG in a bottom-up direction, we add
                # channels from the front, and pop it out when it's no longer
                #  needed by the parent DAG.
                channels_to_add = collections.deque()
                channel_to_add = channel

                while isinstance(channel_to_add, (
                        for_loop.LoopArgument,
                        for_loop.LoopArgumentVariable,
                )):
                    channels_to_add.append(channel_to_add)
                    if isinstance(channel_to_add,
                                  for_loop.LoopArgumentVariable):
                        channel_to_add = channel_to_add.loop_argument
                    else:
                        channel_to_add = channel_to_add.items_or_pipeline_channel

                if isinstance(channel_to_add, dsl.PipelineChannel):
                    channels_to_add.append(channel_to_add)

                if channel.task_name:
                    # The PipelineChannel is produced by a task.

                    upstream_task = pipeline.tasks[channel.task_name]
                    upstream_groups, downstream_groups = (
                        self._get_uncommon_ancestors(
                            task_name_to_parent_groups=task_name_to_parent_groups,
                            group_name_to_parent_groups=group_name_to_parent_groups,
                            task1=upstream_task,
                            task2=task,
                        ))

                    for i, group_name in enumerate(downstream_groups):
                        if i == 0:
                            # If it is the first uncommon downstream group, then
                            # the input comes from the first uncommon upstream
                            # group.
                            producer_task = upstream_groups[0]
                        else:
                            # If not the first downstream group, then the input
                            # is passed down from its ancestor groups so the
                            # upstream group is None.
                            producer_task = None

                        inputs[group_name].add(
                            (channels_to_add[-1], producer_task))

                        if group_name in name_to_for_loop_group:
                            loop_group = name_to_for_loop_group[group_name]

                            # Pop out the last elements from channels_to_add if it
                            # is found in the current (loop) DAG. Downstreams
                            # would only need the more specific versions for it.
                            if channels_to_add[
                                    -1].full_name in loop_group.loop_argument.full_name:
                                channels_to_add.pop()
                                if not channels_to_add:
                                    break

                else:
                    # The PipelineChannel is not produced by a task. It's either
                    # a top-level pipeline input, or a constant value to loop
                    # items.

                    # TODO: revisit if this is correct.
                    if getattr(task, 'is_exit_handler', False):
                        continue

                    # For PipelineChannel as a result of constant value used as
                    # loop items, we have to go from bottom-up because the
                    # PipelineChannel can be originated from the middle a DAG,
                    # which is not needed and visible to its parent DAG.
                    if isinstance(
                            channel,
                        (for_loop.LoopArgument, for_loop.LoopArgumentVariable
                        )) and channel.is_with_items_loop_argument:
                        for group_name in task_name_to_parent_groups[
                                task.name][::-1]:

                            inputs[group_name].add((channels_to_add[0], None))
                            if group_name in name_to_for_loop_group:
                                # for example:
                                #   loop_group.loop_argument.name = 'loop-item-param-1'
                                #   channel.name = 'loop-item-param-1-subvar-a'
                                loop_group = name_to_for_loop_group[group_name]

                                if channels_to_add[
                                        0].full_name in loop_group.loop_argument.full_name:
                                    channels_to_add.popleft()
                                    if not channels_to_add:
                                        break
                    else:
                        # For PipelineChannel from pipeline input, go top-down
                        # just like we do for PipelineChannel produced by a task.
                        for group_name in task_name_to_parent_groups[task.name]:

                            inputs[group_name].add((channels_to_add[-1], None))
                            if group_name in name_to_for_loop_group:
                                loop_group = name_to_for_loop_group[group_name]

                                if channels_to_add[
                                        -1].full_name in loop_group.loop_argument.full_name:
                                    channels_to_add.pop()
                                    if not channels_to_add:
                                        break

        return inputs

    def _get_uncommon_ancestors(
        self,
        task_name_to_parent_groups: Mapping[str, List[_GroupOrTask]],
        group_name_to_parent_groups: Mapping[str, List[tasks_group.TasksGroup]],
        task1: _GroupOrTask,
        task2: _GroupOrTask,
    ) -> Tuple[List[_GroupOrTask], List[_GroupOrTask]]:
        """Gets the unique ancestors between two tasks.

        For example, task1's ancestor groups are [root, G1, G2, G3, task1],
        task2's ancestor groups are [root, G1, G4, task2], then it returns a
        tuple ([G2, G3, task1], [G4, task2]).

        Args:
            task_name_to_parent_groups: The dict of task name to list of parent
                groups.
            group_name_tor_parent_groups: The dict of group name to list of
                parent groups.
            task1: One of the two tasks.
            task2: The other task.

        Returns:
            A tuple which are lists of uncommon ancestors for each task.
        """
        if task1.name in task_name_to_parent_groups:
            task1_groups = task_name_to_parent_groups[task1.name]
        elif task1.name in group_name_to_parent_groups:
            task1_groups = group_name_to_parent_groups[task1.name]
        else:
            raise ValueError(task1.name + ' does not exist.')

        if task2.name in task_name_to_parent_groups:
            task2_groups = task_name_to_parent_groups[task2.name]
        elif task2.name in group_name_to_parent_groups:
            task2_groups = group_name_to_parent_groups[task2.name]
        else:
            raise ValueError(task2.name + ' does not exist.')

        both_groups = [task1_groups, task2_groups]
        common_groups_len = sum(
            1 for x in zip(*both_groups) if x == (x[0],) * len(x))
        group1 = task1_groups[common_groups_len:]
        group2 = task2_groups[common_groups_len:]
        return (group1, group2)

    def _get_dependencies(
        self,
        pipeline: pipeline_context.Pipeline,
        root_group: tasks_group.TasksGroup,
        task_name_to_parent_groups: Mapping[str, List[_GroupOrTask]],
        group_name_to_parent_groups: Mapping[str, List[tasks_group.TasksGroup]],
        group_name_to_group: Mapping[str, tasks_group.TasksGroup],
        condition_channels: Dict[str, dsl.PipelineChannel],
    ) -> Mapping[str, List[_GroupOrTask]]:
        """Gets dependent groups and tasks for all tasks and groups.

        Args:
            pipeline: The instantiated pipeline object.
            root_group: The root group of the pipeline.
            task_name_to_parent_groups: The dict of task name to list of parent
                groups.
            group_name_to_parent_groups: The dict of group name to list of
                parent groups.
            group_name_to_group: The dict of group name to group.
            condition_channels: The dict of task name to a set of pipeline
                channels referenced by its parent condition groups.

        Returns:
            A Mapping where key is group/task name, value is a list of dependent
            groups/tasks. The dependencies are calculated in the following way:
            if task2 depends on task1, and their ancestors are
            [root, G1, G2, task1] and [root, G1, G3, G4, task2], then G3 is
            dependent on G2. Basically dependency only exists in the first
            uncommon ancesters in their ancesters chain. Only sibling
            groups/tasks can have dependencies.

        Raises:
            RuntimeError: if a task depends on a task inside a condition or loop
                group.
        """
        dependencies = collections.defaultdict(set)
        for task in pipeline.tasks.values():
            upstream_task_names = set()
            task_condition_inputs = list(condition_channels[task.name])
            for channel in task.channel_inputs + task_condition_inputs:
                if channel.task_name:
                    upstream_task_names.add(channel.task_name)
            upstream_task_names |= set(task.dependent_tasks)

            for upstream_task_name in upstream_task_names:
                # the dependent op could be either a BaseOp or an opsgroup
                if upstream_task_name in pipeline.tasks:
                    upstream_task = pipeline.tasks[upstream_task_name]
                elif upstream_task_name in group_name_to_group:
                    upstream_task = group_name_to_group[upstream_task_name]
                else:
                    raise ValueError(
                        f'Compiler cannot find task: {upstream_task_name}.')

                upstream_groups, downstream_groups = self._get_uncommon_ancestors(
                    task_name_to_parent_groups=task_name_to_parent_groups,
                    group_name_to_parent_groups=group_name_to_parent_groups,
                    task1=upstream_task,
                    task2=task,
                )

                # If a task depends on a condition group or a loop group, it
                # must explicitly dependent on a task inside the group. This
                # should not be allowed, because it leads to ambiguous
                # expectations for runtime behaviors.
                dependent_group = group_name_to_group.get(
                    upstream_groups[0], None)
                if isinstance(dependent_group,
                              (tasks_group.Condition, tasks_group.ParallelFor)):
                    raise RuntimeError(
                        f'Task {task.name} cannot dependent on any task inside'
                        f' the group: {upstream_groups[0]}.')

                dependencies[downstream_groups[0]].add(upstream_groups[0])

        return dependencies

    def _build_spec_by_group(
        self,
        pipeline_spec: pipeline_spec_pb2.PipelineSpec,
        deployment_config: pipeline_spec_pb2.PipelineDeploymentConfig,
        group: tasks_group.TasksGroup,
        inputs: Mapping[str, List[Tuple[dsl.PipelineChannel, str]]],
        dependencies: Dict[str, List[_GroupOrTask]],
        rootgroup_name: str,
        task_name_to_parent_groups: Mapping[str, List[_GroupOrTask]],
        group_name_to_parent_groups: Mapping[str, List[tasks_group.TasksGroup]],
        name_to_for_loop_group: Mapping[str, dsl.ParallelFor],
    ) -> None:
        """Generates IR spec given a TasksGroup.

        Args:
            pipeline_spec: The pipeline_spec to update in place.
            deployment_config: The deployment_config to hold all executors. The
                spec is updated in place.
            group: The TasksGroup to generate spec for.
            inputs: The inputs dictionary. The keys are group/task names and the
                values are lists of tuples (channel, producing_task_name).
            dependencies: The group dependencies dictionary. The keys are group
                or task names, and the values are lists of dependent groups or
                tasks.
            rootgroup_name: The name of the group root. Used to determine whether
                the component spec for the current group should be the root dag.
            task_name_to_parent_groups: The dict of task name to parent groups.
                Key is task name. Value is a list of ancestor groups including
                the task itself. The list of a given task is sorted in a way that
                the farthest group is the first and the task itself is the last.
            group_name_to_parent_groups: The dict of group name to parent groups.
                Key is the group name. Value is a list of ancestor groups
                including the group itself. The list of a given group is sorted
                in a way that the farthest group is the first and the group
                itself is the last.
            name_to_for_loop_group: The dict of for loop group name to loop
                group.
        """
        group_component_name = component_utils.sanitize_component_name(
            group.name)

        if group.name == rootgroup_name:
            group_component_spec = pipeline_spec.root
        else:
            group_component_spec = pipeline_spec.components[
                group_component_name]

        task_name_to_task_spec = {}
        task_name_to_component_spec = {}

        # Generate task specs and component specs for the dag.
        subgroups = group.groups + group.tasks
        for subgroup in subgroups:
            subgroup_task_spec = getattr(subgroup, 'task_spec',
                                         pipeline_spec_pb2.PipelineTaskSpec())
            subgroup_component_spec = getattr(subgroup, 'component_spec',
                                              pipeline_spec_pb2.ComponentSpec())

            display_name = getattr(subgroup, 'display_name', None)
            subgroup_task_spec.task_info.name = display_name or subgroup.name

            is_recursive_subgroup = (
                isinstance(subgroup, dsl.OpsGroup) and subgroup.recursive_ref)
            if is_recursive_subgroup:
                raise NotImplementedError(
                    'Recursive subgroup is not supported in v2 yet.')
            else:
                subgroup_key = subgroup.name

            # human_name exists for ops only, and is used to de-dupe component spec.
            subgroup_component_name = (
                subgroup_task_spec.component_ref.name or
                dsl_utils.sanitize_component_name(
                    getattr(subgroup, 'human_name', subgroup_key)))
            subgroup_task_spec.component_ref.name = subgroup_component_name

            if isinstance(subgroup, dsl.OpsGroup) and subgroup.type == 'graph':
                raise NotImplementedError(
                    'dsl.graph_component is not yet supported in KFP v2 compiler.'
                )

            if isinstance(subgroup, dsl.ContainerOp):
                if hasattr(subgroup, 'importer_spec'):
                    importer_comp_name = subgroup.task_spec.component_ref.name
                    importer_exec_label = subgroup.component_spec.executor_label
                    group_component_spec.dag.tasks[subgroup.name].CopyFrom(
                        subgroup.task_spec)
                    pipeline_spec.components[importer_comp_name].CopyFrom(
                        subgroup.component_spec)
                    deployment_config.executors[
                        importer_exec_label].importer.CopyFrom(
                            subgroup.importer_spec)
                else:
                    for input_spec in subgroup._metadata.inputs or []:
                        if type_utils.is_task_final_status_type(
                                input_spec.type):
                            raise ValueError(
                                f'{task_final_status.PipelineTaskFinalStatus.__name__}'
                                ' can only be used in an exit task.')

                # Task level caching option.
                subgroup.task_spec.caching_options.enable_cache = subgroup.enable_caching
                if subgroup.num_retries or subgroup.backoff_duration or subgroup.backoff_factor or subgroup.backoff_max_duration:

                    if subgroup.retry_policy is not None:
                        warnings.warn(
                            "'retry_policy' is ignored when compiling to IR using the v2 compiler.",
                            compiler_utils.NoOpWarning)

                    retry_policy_proto = compiler_utils.make_retry_policy_proto(
                        max_retry_count=subgroup.num_retries,
                        backoff_duration=subgroup.backoff_duration,
                        backoff_factor=subgroup.backoff_factor,
                        backoff_max_duration=subgroup.backoff_max_duration,
                    )
                    subgroup.task_spec.retry_policy.CopyFrom(retry_policy_proto)

            subgroup_inputs = inputs.get(subgroup.name, [])
            subgroup_channels = [channel for channel, _ in subgroup_inputs]

            subgroup_component_name = (
                component_utils.sanitize_component_name(subgroup.name))

            tasks_in_current_dag = [
                component_utils.sanitize_task_name(subgroup.name)
                for subgroup in subgroups
            ]
            input_parameters_in_current_dag = [
                input_name for input_name in
                group_component_spec.input_definitions.parameters
            ]
            input_artifacts_in_current_dag = [
                input_name for input_name in
                group_component_spec.input_definitions.artifacts
            ]
            is_parent_component_root = (
                group_component_spec == pipeline_spec.root)

            if isinstance(subgroup, pipeline_task.PipelineTask):

                subgroup_task_spec = builder.build_task_spec_for_task(
                    task=subgroup,
                    parent_component_inputs=group_component_spec
                    .input_definitions,
                    tasks_in_current_dag=tasks_in_current_dag,
                    input_parameters_in_current_dag=input_parameters_in_current_dag,
                    input_artifacts_in_current_dag=input_artifacts_in_current_dag,
                )
                task_name_to_task_spec[subgroup.name] = subgroup_task_spec

            if isinstance(subgroup, dsl.ParallelFor):
                # "Punch the hole", adding additional inputs (other than loop
                # arguments which will be handled separately) needed by its
                # subgroups or tasks.
                loop_subgroup_channels = []

                for channel in subgroup_channels:
                    # Skip 'withItems' loop arguments if it's from an inner loop.
                    if isinstance(
                            channel,
                        (for_loop.LoopArgument, for_loop.LoopArgumentVariable
                        )) and channel.is_with_items_loop_argument:
                        withitems_loop_arg_found_in_self_or_upstream = False
                        for group_name in group_name_to_parent_groups[
                                subgroup.name][::-1]:
                            if group_name in name_to_for_loop_group:
                                loop_group = name_to_for_loop_group[group_name]
                                if channel.name in loop_group.loop_argument.name:
                                    withitems_loop_arg_found_in_self_or_upstream = True
                                    break
                        if not withitems_loop_arg_found_in_self_or_upstream:
                            continue
                    loop_subgroup_channels.append(channel)

                if subgroup.items_is_pipeline_channel:
                    # This loop_argument is based on a pipeline channel, i.e.,
                    # rather than a static list, it is either the output of
                    # another task or an input as global pipeline parameters.
                    loop_subgroup_channels.append(
                        subgroup.loop_argument.items_or_pipeline_channel)

                loop_subgroup_channels.append(subgroup.loop_argument)

                subgroup_component_spec = builder.build_component_spec_for_group(
                    pipeline_channels=loop_subgroup_channels,
                    is_root_group=False,
                )

                subgroup_task_spec = builder.build_task_spec_for_group(
                    group=subgroup,
                    pipeline_channels=loop_subgroup_channels,
                    tasks_in_current_dag=tasks_in_current_dag,
                    is_parent_component_root=is_parent_component_root,
                )

            elif isinstance(subgroup, dsl.Condition):

                    # If the loop arguments itself is a loop arguments variable, handle
                    # the subvar name.
                    loop_args_name, subvar_name = (
                        dsl_component_spec._exclude_loop_arguments_variables(
                            subgroup.loop_args.items_or_pipeline_param))
                    if subvar_name:
                        subgroup_task_spec.inputs.parameters[
                            input_parameter_name].parameter_expression_selector = (
                                'parseJson(string_value)["{}"]'.format(
                                    subvar_name))
                        subgroup_task_spec.inputs.parameters[
                            input_parameter_name].component_input_parameter = (
                                dsl_component_spec
                                .additional_input_name_for_pipelineparam(
                                    loop_args_name))

                else:
                    input_parameter_name = (
                        dsl_component_spec
                        .additional_input_name_for_pipelineparam(
                            subgroup.loop_args.full_name))
                    raw_values = subgroup.loop_args.to_list_for_task_yaml()

                    dsl_component_spec.pop_input_from_task_spec(
                        task_spec=subgroup_task_spec,
                        input_name=input_parameter_name)

                    subgroup_task_spec.parameter_iterator.items.raw = json.dumps(
                        raw_values, sort_keys=True)
                    subgroup_task_spec.parameter_iterator.item_input = (
                        input_parameter_name)
                if subgroup.parallelism > 0:
                    subgroup_task_spec.iterator_policy.parallelism_limit = (
                        subgroup.parallelism)

            if isinstance(subgroup, dsl.Condition):

                # "punch the hole", adding inputs needed by its subgroup or tasks.
                condition_params = list(subgroup_params)
                for param in [
                        subgroup.condition.operand1,
                        subgroup.condition.operand2,
                ]:
                    if isinstance(operand, dsl.PipelineChannel):
                        condition_subgroup_channels.append(operand)

                subgroup_component_spec = builder.build_component_spec_for_group(
                    pipeline_channels=condition_subgroup_channels,
                    is_root_group=False,
                )

                subgroup_task_spec = builder.build_task_spec_for_group(
                    group=subgroup,
                    pipeline_channels=condition_subgroup_channels,
                    tasks_in_current_dag=tasks_in_current_dag,
                    is_parent_component_root=is_parent_component_root,
                )

            elif isinstance(subgroup, dsl.ExitHandler):

                subgroup_component_spec = builder.build_component_spec_for_group(
                    pipeline_channels=subgroup_channels,
                    is_root_group=False,
                )

                subgroup_task_spec = builder.build_task_spec_for_group(
                    group=subgroup,
                    pipeline_channels=subgroup_channels,
                    tasks_in_current_dag=tasks_in_current_dag,
                    is_parent_component_root=is_parent_component_root,
                )

            else:
                raise RuntimeError(
                    f'Unexpected task/group type: Got {subgroup} of type '
                    f'{type(subgroup)}.')

            # Generate dependencies section for this task.
            if dependencies.get(subgroup.name, None):
                group_dependencies = list(dependencies[subgroup.name])
                group_dependencies.sort()
                subgroup_task_spec.dependent_tasks.extend([
                    component_utils.sanitize_task_name(dep)
                    for dep in group_dependencies
                ])

            # Add component spec if not exists
            if subgroup_component_name not in pipeline_spec.components:
                pipeline_spec.components[subgroup_component_name].CopyFrom(
                    subgroup_component_spec)

            # Add task spec
            group_component_spec.dag.tasks[subgroup.name].CopyFrom(
                subgroup_task_spec)

        pipeline_spec.deployment_spec.update(
            json_format.MessageToDict(deployment_config))

        # Surface metrics outputs to the top.
        builder.populate_metrics_in_dag_outputs(
            tasks=group.tasks,
            task_name_to_parent_groups=task_name_to_parent_groups,
            task_name_to_task_spec=task_name_to_task_spec,
            task_name_to_component_spec=task_name_to_component_spec,
            pipeline_spec=pipeline_spec,
        )

    def _create_pipeline_spec(
        self,
        args: List[dsl.PipelineParam],
        pipeline: dsl.Pipeline,
    ) -> pipeline_spec_pb2.PipelineSpec:
        """Creates the pipeline spec object.

        Args:
          args: The list of pipeline arguments.
          pipeline: The instantiated pipeline object.

        Returns:
          A PipelineSpec proto representing the compiled pipeline.

        Raises:
          NotImplementedError if the argument is of unsupported types.
        """
        compiler_utils.validate_pipeline_name(pipeline.name)

        deployment_config = pipeline_spec_pb2.PipelineDeploymentConfig()
        pipeline_spec = pipeline_spec_pb2.PipelineSpec()

        pipeline_spec.pipeline_info.name = pipeline.name
        pipeline_spec.sdk_version = 'kfp-{}'.format(kfp.__version__)
        # Schema version 2.0.0 is required for kfp-pipeline-spec>0.1.3.1
        pipeline_spec.schema_version = '2.0.0'

        dsl_component_spec.build_component_inputs_spec(
            component_spec=pipeline_spec.root,
            pipeline_params=args,
            is_root_component=True)

        root_group = pipeline.groups[0]
        opsgroups = self._get_groups(root_group)
        op_name_to_parent_groups = self._get_groups_for_ops(root_group)
        opgroup_name_to_parent_groups = self._get_groups_for_opsgroups(
            root_group)

        condition_params = self._get_condition_params_for_ops(root_group)
        op_name_to_for_loop_op = self._get_for_loop_ops(root_group)

        inputs, _ = self._get_inputs_outputs(
            pipeline=pipeline,
            args=args,
            root_group=root_group,
            op_groups=op_name_to_parent_groups,
            opsgroup_groups=opgroup_name_to_parent_groups,
            condition_params=condition_params,
            op_name_to_for_loop_op=op_name_to_for_loop_op,
        )

        dependencies = self._get_dependencies(
            pipeline=pipeline,
            root_group=root_group,
            op_groups=op_name_to_parent_groups,
            opsgroups_groups=opgroup_name_to_parent_groups,
            opsgroups=opsgroups,
            condition_params=condition_params,
        )

        for opsgroup_name in opsgroups.keys():
            self._group_to_dag_spec(
                group=opsgroups[opsgroup_name],
                inputs=inputs,
                dependencies=dependencies,
                pipeline_spec=pipeline_spec,
                deployment_config=deployment_config,
                rootgroup_name=root_group.name,
                op_to_parent_groups=op_name_to_parent_groups,
                opgroup_to_parent_groups=opgroup_name_to_parent_groups,
                op_name_to_for_loop_op=op_name_to_for_loop_op,
            )

        # Exit Handler
        if pipeline.groups[0].groups:
            first_group = pipeline.groups[0].groups[0]
            if first_group.type == 'exit_handler':
                exit_handler_op = first_group.exit_op

                # Add exit op task spec
                task_name = exit_handler_op.name
                display_name = exit_handler_op.display_name

                exit_handler_op.task_spec.task_info.name = (
                    display_name or task_name)
                for input_spec in exit_handler_op._metadata.inputs or []:
                    if type_utils.is_task_final_status_type(input_spec.type):
                        exit_handler_op.task_spec.inputs.parameters[
                            input_spec.name].task_final_status.producer_task = (
                                first_group.name)

                exit_handler_op.task_spec.dependent_tasks.extend(
                    pipeline_spec.root.dag.tasks.keys())
                exit_handler_op.task_spec.trigger_policy.strategy = (
                    pipeline_spec_pb2.PipelineTaskSpec.TriggerPolicy
                    .TriggerStrategy.ALL_UPSTREAM_TASKS_COMPLETED)
                pipeline_spec.root.dag.tasks[task_name].CopyFrom(
                    exit_handler_op.task_spec)

                # Add exit op component spec if it does not exist.
                component_name = exit_handler_op.task_spec.component_ref.name
                if component_name not in pipeline_spec.components:
                    pipeline_spec.components[component_name].CopyFrom(
                        exit_handler_op.component_spec)

                # Add exit op executor spec if it does not exist.
                executor_label = exit_handler_op.component_spec.executor_label
                if executor_label not in deployment_config.executors:
                    deployment_config.executors[
                        executor_label].container.CopyFrom(
                            exit_handler_op.container_spec)
                    pipeline_spec.deployment_spec.update(
                        json_format.MessageToDict(deployment_config))

        return pipeline_spec

    def _validate_exit_handler(self, pipeline):
        """Makes sure there is only one global exit handler.

        This is temporary to be compatible with KFP v1.
        """

        def _validate_exit_handler_helper(group, exiting_op_names,
                                          handler_exists):
            if group.type == 'exit_handler':
                if handler_exists or len(exiting_op_names) > 1:
                    raise ValueError(
                        'Only one global exit_handler is allowed and all ops need to be included.'
                    )
                handler_exists = True

            if group.ops:
                exiting_op_names.extend([x.name for x in group.ops])

            for g in group.groups:
                _validate_exit_handler_helper(g, exiting_op_names,
                                              handler_exists)

        return _validate_exit_handler_helper(pipeline.groups[0], [], False)

    # TODO: Sanitizing beforehand, so that we don't need to sanitize here.
    def _sanitize_and_inject_artifact(self, pipeline: dsl.Pipeline) -> None:
        """Sanitize operator/param names and inject pipeline artifact
        location."""

        # Sanitize operator names and param names
        sanitized_ops = {}

        for op in pipeline.ops.values():
            sanitized_name = sanitize_k8s_name(op.name)
            op.name = sanitized_name
            for param in op.outputs.values():
                param.name = sanitize_k8s_name(param.name, True)
                if param.op_name:
                    param.op_name = sanitize_k8s_name(param.op_name)
            if op.output is not None and not isinstance(
                    op.output, dsl._container_op._MultipleOutputsError):
                op.output.name = sanitize_k8s_name(op.output.name, True)
                op.output.op_name = sanitize_k8s_name(op.output.op_name)
            if op.dependent_names:
                op.dependent_names = [
                    sanitize_k8s_name(name) for name in op.dependent_names
                ]
            if isinstance(op, dsl.ContainerOp) and op.file_outputs is not None:
                sanitized_file_outputs = {}
                for key in op.file_outputs.keys():
                    sanitized_file_outputs[sanitize_k8s_name(
                        key, True)] = op.file_outputs[key]
                op.file_outputs = sanitized_file_outputs
            elif isinstance(
                    op, dsl.ResourceOp) and op.attribute_outputs is not None:
                sanitized_attribute_outputs = {}
                for key in op.attribute_outputs.keys():
                    sanitized_attribute_outputs[sanitize_k8s_name(key, True)] = \
                      op.attribute_outputs[key]
                op.attribute_outputs = sanitized_attribute_outputs
            if isinstance(op, dsl.ContainerOp):
                if op.input_artifact_paths:
                    op.input_artifact_paths = {
                        sanitize_k8s_name(key, True): value
                        for key, value in op.input_artifact_paths.items()
                    }
                if op.artifact_arguments:
                    op.artifact_arguments = {
                        sanitize_k8s_name(key, True): value
                        for key, value in op.artifact_arguments.items()
                    }
            sanitized_ops[sanitized_name] = op
        pipeline.ops = sanitized_ops

    def _create_pipeline_v2(
        self,
        pipeline_func: Callable[..., Any],
        pipeline_name: Optional[str] = None,
        pipeline_parameters_override: Optional[Mapping[str, Any]] = None,
    ) -> pipeline_spec_pb2.PipelineJob:
        """Creates a pipeline instance and constructs the pipeline spec from
        it.

        Args:
          pipeline_func: Pipeline function with @dsl.pipeline decorator.
          pipeline_name: The name of the pipeline. Optional.
          pipeline_parameters_override: The mapping from parameter names to values.
            Optional.

        Returns:
          A PipelineJob proto representing the compiled pipeline.
        """

        # Create the arg list with no default values and call pipeline function.
        # Assign type information to the PipelineParam
        pipeline_meta = component_factory.extract_component_interface(
            pipeline_func)
        pipeline_name = pipeline_name or pipeline_meta.name

        pipeline_root = getattr(pipeline_func, 'pipeline_root', None)

        args_list = []
        signature = inspect.signature(pipeline_func)
        for arg_name in signature.parameters:
            arg_type = None
            for pipeline_input in pipeline_meta.inputs or []:
                if arg_name == pipeline_input.name:
                    arg_type = pipeline_input.type
                    break
            if not type_utils.is_parameter_type(arg_type):
                raise TypeError(
                    'The pipeline argument "{arg_name}" is viewed as an artifact'
                    ' due to its type "{arg_type}". And we currently do not '
                    'support passing artifacts as pipeline inputs. Consider type'
                    ' annotating the argument with a primitive type, such as '
                    '"str", "int", "float", "bool", "dict", and "list".'.format(
                        arg_name=arg_name, arg_type=arg_type))
            args_list.append(
                dsl.PipelineParam(
                    sanitize_k8s_name(arg_name, True), param_type=arg_type))

        with dsl.Pipeline(pipeline_name) as dsl_pipeline:
            pipeline_func(*args_list)

        if not dsl_pipeline.ops:
            raise ValueError('Task is missing from pipeline.')

        self._validate_exit_handler(dsl_pipeline)
        self._sanitize_and_inject_artifact(dsl_pipeline)

        # Fill in the default values.
        args_list_with_defaults = []
        if pipeline_meta.inputs:
            args_list_with_defaults = [
                dsl.PipelineParam(
                    sanitize_k8s_name(input_spec.name, True),
                    param_type=input_spec.type,
                    value=input_spec.default)
                for input_spec in pipeline_meta.inputs
            ]

        # Making the pipeline group name unique to prevent name clashes with templates
        pipeline_group = dsl_pipeline.groups[0]
        temp_pipeline_group_name = uuid.uuid4().hex
        pipeline_group.name = temp_pipeline_group_name

        pipeline_spec = self._create_pipeline_spec(
            args_list_with_defaults,
            dsl_pipeline,
        )

        pipeline_parameters = {
            param.name: param for param in args_list_with_defaults
        }
        # Update pipeline parameters override if there were any.
        pipeline_parameters_override = pipeline_parameters_override or {}
        for k, v in pipeline_parameters_override.items():
            if k not in pipeline_parameters:
                raise ValueError(
                    'Pipeline parameter {} does not match any known '
                    'pipeline argument.'.format(k))
            pipeline_parameters[k].value = v

        runtime_config = compiler_utils.build_runtime_config_spec(
            output_directory=pipeline_root,
            pipeline_parameters=pipeline_parameters)
        pipeline_job = pipeline_spec_pb2.PipelineJob(
            runtime_config=runtime_config)
        pipeline_job.pipeline_spec.update(
            json_format.MessageToDict(pipeline_spec))

        return pipeline_job

    def compile(self,
                pipeline_func: Callable[..., Any],
                package_path: str,
                pipeline_name: Optional[str] = None,
                pipeline_parameters: Optional[Mapping[str, Any]] = None,
                type_check: bool = True) -> None:
        """Compile the given pipeline function into pipeline job json.

        Args:
          pipeline_func: Pipeline function with @dsl.pipeline decorator.
          package_path: The output pipeline job .json file path. for example,
            "~/pipeline_job.json"
          pipeline_name: The name of the pipeline. Optional.
          pipeline_parameters: The mapping from parameter names to values. Optional.
          type_check: Whether to enable the type check or not, default: True.
        """
        warnings.warn(
            'APIs imported from the v1 namespace (e.g. kfp.dsl, kfp.components, '
            'etc) will not be supported by the v2 compiler since v2.0.0',
            category=FutureWarning,
        )

        type_check_old_value = kfp.TYPE_CHECK
        compiling_for_v2_old_value = kfp.COMPILING_FOR_V2
        try:
            kfp.TYPE_CHECK = type_check
            kfp.COMPILING_FOR_V2 = True
            pipeline_job = self._create_pipeline_v2(
                pipeline_func=pipeline_func,
                pipeline_name=pipeline_name,
                pipeline_parameters_override=pipeline_parameters)
            self._write_pipeline(pipeline_job, package_path)
        finally:
            kfp.TYPE_CHECK = type_check_old_value
            kfp.COMPILING_FOR_V2 = compiling_for_v2_old_value

    def _write_pipeline(self, pipeline_job: pipeline_spec_pb2.PipelineJob,
                        output_path: str) -> None:
        """Dump pipeline spec into json file.

        Args:
          pipeline_job: IR pipeline job spec.
          ouput_path: The file path to be written.

        Raises:
          ValueError: if the specified output path doesn't end with the acceptable
          extentions.
        """
        json_text = json_format.MessageToJson(pipeline_job, sort_keys=True)

        if output_path.endswith('.json'):
            with open(output_path, 'w') as json_file:
                json_file.write(json_text)
        else:
            raise ValueError(
                'The output path {} should ends with ".json".'.format(
                    output_path))
