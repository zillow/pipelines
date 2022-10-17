// Copyright 2019-2021 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
import { Handler } from 'express';
import * as k8sHelper from '../k8s-helper';
import { parseError } from '../utils';


/**
 * workflowDeleteHandler deletes Argo Workflow
 */
export const workflowDeleteHandler: Handler = async (req, res) => {
  const { workflowName, namespace } = req.query;
  if (!workflowName) {
    res.status(400).send('workflowName argument is required');
    return;
  }
  if (!namespace) {
    res.status(400).send('namespace argument is required');
    return;
  }


  try {
    await k8sHelper.deleteArgoWorkflow(workflowName, namespace);
    res.send('Workflow deleted.');
  } catch (err) {
    const details = await parseError(err);
    console.error(`Failed to delete workflow ${workflowName} ${namespace}: ${details.message}`, details.additionalInfo);
    res.status(500).send(`Failed to delete ${workflowName} ${namespace}: ${details.message}`);
  }
};


/**
 * workflowStopHandler stops an Argo Workflow
 */
export const workflowStopHandler: Handler = async (req, res) => {
  const { workflowName, namespace } = req.query;
  if (!workflowName) {
    res.status(400).send('workflowName argument is required');
    return;
  }
  if (!namespace) {
    res.status(400).send('namespace argument is required');
    return;
  }


  try {
    await k8sHelper.deleteArgoWorkflow(workflowName, namespace);
    res.send('Workflow deleted.');
  } catch (err) {
    const details = await parseError(err);
    console.error(`Failed to delete workflow ${workflowName} ${namespace}: ${details.message}`, details.additionalInfo);
    res.status(500).send(`Failed to delete ${workflowName} ${namespace}: ${details.message}`);
  }
}