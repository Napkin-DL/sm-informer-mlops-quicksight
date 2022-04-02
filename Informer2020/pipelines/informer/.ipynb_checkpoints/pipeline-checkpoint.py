"""Example workflow pipeline script for abalone pipeline.

                                               . -RegisterModel
                                              .
    Process-> Train -> Evaluate -> Condition .
                                              .
                                               . -(stop)

Implements a get_pipeline(**kwargs) method.
"""

# sagemaker-experiments

import os

import boto3
import sagemaker
import sagemaker.session

import datetime
import glob
import os
import time
import warnings

from smexperiments.experiment import Experiment
from smexperiments.trial import Trial


import shutil

import boto3
import numpy as np
import pandas as pd
import subprocess

# from tqdm import tqdm
from time import strftime

from sagemaker import get_execution_role
from sagemaker.pytorch import PyTorch

from sagemaker.inputs import TrainingInput

from sagemaker.model_metrics import (
    MetricsSource,
    ModelMetrics,
)
from sagemaker.processing import (
    ProcessingInput,
    ProcessingOutput,
    ScriptProcessor,
)
from sagemaker.sklearn.processing import SKLearnProcessor
from sagemaker.workflow.conditions import ConditionLessThanOrEqualTo
from sagemaker.workflow.condition_step import (
    ConditionStep,
    JsonGet,
)
from sagemaker.workflow.parameters import (
    ParameterInteger,
    ParameterString,
    ParameterFloat
)
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import (
    ProcessingStep,
    TrainingStep,
)
from sagemaker.workflow.step_collections import RegisterModel
from sagemaker.workflow.steps import CacheConfig

BASE_DIR = os.path.dirname(os.path.realpath(__file__))
SOURCE_DIR = 'Informer2020'


def get_sagemaker_client(region):
     """Gets the sagemaker client.

        Args:
            region: the aws region to start the session
            default_bucket: the bucket to use for storing the artifacts

        Returns:
            `sagemaker.session.Session instance
        """
     boto_session = boto3.Session(region_name=region)
     sagemaker_client = boto_session.client("sagemaker")
     return sagemaker_client


def get_session(region):
    """Gets the sagemaker session based on the region.

    Args:
        region: the aws region to start the session
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        `sagemaker.session.Session instance
    """

    boto_session = boto3.Session(region_name=region)

    sagemaker_client = boto_session.client("sagemaker")
    runtime_client = boto_session.client("sagemaker-runtime")
    return sagemaker.session.Session(
        boto_session=boto_session,
        sagemaker_client=sagemaker_client,
        sagemaker_runtime_client=runtime_client,
#         default_bucket=default_bucket,
    )

def get_pipeline_custom_tags(new_tags, region, sagemaker_project_arn=None):
    try:
        sm_client = get_sagemaker_client(region)
        response = sm_client.list_tags(
            ResourceArn=sagemaker_project_arn)
        project_tags = response["Tags"]
        for project_tag in project_tags:
            new_tags.append(project_tag)
    except Exception as e:
        print(f"Error getting project tags: {e}")
    return new_tags


def create_experiment(experiment_name):
    try:
        sm_experiment = Experiment.load(experiment_name)
    except:
        sm_experiment = Experiment.create(experiment_name=experiment_name,
                                          tags=[
                                              {
                                                  'Key': 'modelname',
                                                  'Value': 'informer'
                                              },
                                          ])


def create_trial(experiment_name, set_param, i_type, i_cnt, spot):
    create_date = strftime("%m%d-%H%M%s")
    
    algo = 'dp'
    
    spot = 's' if spot else 'd'
    i_tag = 'test'
    if i_type == 'ml.p3.16xlarge':
        i_tag = 'p3'
    elif i_type == 'ml.p3dn.24xlarge':
        i_tag = 'p3dn'
    elif i_type == 'ml.p4d.24xlarge':
        i_tag = 'p4d'    
        
    trial = "-".join([i_tag,str(i_cnt),algo, spot])
       
    sm_trial = Trial.create(trial_name=f'{experiment_name}-{trial}-{create_date}',
                            experiment_name=experiment_name)

    job_name = f'{sm_trial.trial_name}'
    return job_name



def get_pipeline(
    region,
    sagemaker_project_arn=None,
    role=None,
    default_bucket=None,
    model_package_group_name="TimeSeries-Group",
    pipeline_name="InformerPipeline",
    base_job_prefix="informer",
):
    """Gets a SageMaker ML Pipeline instance working with on ts data.

    Args:
        region: AWS region to create and run the pipeline.
        role: IAM role to create and run steps and pipeline.
        default_bucket: the bucket to use for storing the artifacts

    Returns:
        an instance of a pipeline
    """
    sagemaker_session = get_session(region)
    default_bucket = sagemaker_session.default_bucket()
    if role is None:
        role = sagemaker.session.get_execution_role(sagemaker_session)

        
    code_repo = f"s3://{default_bucket}/{SOURCE_DIR}"
    cmd = ["aws", "s3", "sync", "--quiet", SOURCE_DIR, code_repo]
    print(f"Syncing files from {SOURCE_DIR} to {code_repo}")
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    p.wait()
    
 
    train_instance_param = ParameterString(
        name="TrainingInstance",
#         default_value="ml.p3.16xlarge",
        default_value='ml.c5.4xlarge'
    )

    train_count_param = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    source_dir = ParameterString(
        name="Informer2020",
        default_value=SOURCE_DIR,
    )


    code_location = f's3://{default_bucket}/sm_codes'
    output_path = f's3://{default_bucket}/poc_informer/output' 
    checkpoint_s3_bucket = f's3://{default_bucket}/checkpoints'

    metric_definitions = [
        {'Name': 'Epoch', 'Regex': 'Epoch: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?),'},
        {'Name': 'train_loss', 'Regex': 'Train Loss: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?),'},
        {'Name': 'valid_loss', 'Regex': 'Valid Loss: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?),'},
        {'Name': 'test_loss', 'Regex': 'Test Loss: ([-+]?[0-9]*[.]?[0-9]+([eE][-+]?[0-9]+)?),'},
    ]
    
    hyperparameters = {
            'model' : 'informer', # model of experiment, options: [informer, informerstack, informerlight(TBD)]
            'data' : 'ETTh1', # data
            'root_path' : 'ETT-small/', # root path of data file
            'data_path' : 'ETTh1.csv', # data file
            'features' : 'M', # forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate
            'target' : 'OT', # target feature in S or MS task
            'freq' : 'h', # freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h
            'checkpoints' : 'informer_checkpoints', # location of model checkpoints

            'seq_len' : 96, # input sequence length of Informer encoder
            'label_len' : 48, # start token length of Informer decoder
            'pred_len' : 24, # prediction sequence length
            # Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

            'enc_in' : 7, # encoder input size
            'dec_in' : 7, # decoder input size
            'c_out' : 7, # output size
            'factor' : 5, # probsparse attn factor
            'd_model' : 512, # dimension of model
            'n_heads' : 8, # num of heads
            'e_layers' : 2, # num of encoder layers
            'd_layers' : 1, # num of decoder layers
            'd_ff' : 2048, # dimension of fcn in model
            'dropout' : 0.05, # dropout
            'attn' : 'prob', # attention used in encoder, options:[prob, full]
            'embed' : 'timeF', # time features encoding, options:[timeF, fixed, learned]
            'activation' : 'gelu', # activation
            'distil' : True, # whether to use distilling in encoder
            'output_attention' : False, # whether to output attention in ecoder
            'mix' : True,
            'padding' : 0,
            'freq' : 'h',
            'do_predict' : True,
            'batch_size' : 32,
            'learning_rate' : 0.0001,
            'loss' : 'mse',
            'lradj' : 'type1',
            'use_amp' : False, # whether to use automatic mixed precision training

            'num_workers' : 0,
            'itr' : 1,
            'train_epochs' : 1,  ## Training epochs
            'patience' : 3,
            'des' : 'exp',
            'use_multi_gpu' : True
        }

    experiment_name = 'informer-poc-exp1'
    instance_type = train_instance_param.default_value
#     instance_count = 1
    do_spot_training = True
    max_wait = None
    max_run = 3*60*60
    
    image_uri = None
    train_job_name = 'sagemaker'
    
    prefix = 'ETDataset'


    s3_data_path = f's3://{default_bucket}/{prefix}'


    train_job_name = 'informer-dist'
    distribution = {}

    if instance_type in ['ml.p3.16xlarge', 'ml.p3dn.24xlarge', 'ml.p4d.24xlarge', 'local_gpu']:
        distribution["smdistributed"]={ 
                            "dataparallel": {
                                "enabled": True
                            }
                    }
    else:
        distribution = None

    if do_spot_training:
        max_wait = max_run
    

    # all input configurations, parameters, and metrics specified in estimator 
    # definition are automatically tracked
    estimator = PyTorch(
        entry_point='main_informer.py',
        source_dir=source_dir,
        role=role,
        sagemaker_session=sagemaker_session,
        framework_version='1.8.1',
        py_version='py36',
        instance_count=train_count_param,    ## Parameter 값으로 변경
        instance_type=train_instance_param,  ## Parameter 값으로 변경
        volume_size=256,
        code_location = code_location,
        output_path=output_path,
        hyperparameters=hyperparameters,
        distribution=distribution,
        metric_definitions=metric_definitions,
        max_run=max_run,
        checkpoint_s3_uri=checkpoint_s3_bucket,
        use_spot_instances=do_spot_training,  # spot instance 활용
        max_wait=max_wait,
        base_job_name=f"informer-train",
    )
    
    cache_config = None
    

#     cache_config = CacheConfig(enable_caching=True, 
#                                expire_after="7d")
    
    training_step = TrainingStep(
        name="InformerTrain",
        estimator=estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=s3_data_path
            )
        },
        cache_config=cache_config
    )

    sklearn_processor = SKLearnProcessor(
        framework_version="0.23-1",
        instance_type="ml.c4.xlarge",
        instance_count=1,
        base_job_name=f"GeneratingReport",  # choose any name
        sagemaker_session=sagemaker_session,
        role=role,
    )

    
    model_input = ProcessingInput(
                            source=training_step.properties.ModelArtifacts.S3ModelArtifacts,
                            destination="/opt/ml/processing/model",
                        )    


    test_report = PropertyFile(
        name="TestReport",
        output_name="result",
        path="test_report.json",
    )

    postprocessing_step = ProcessingStep(
        name="PostProcessingforInformer",  # choose any name
        processor=sklearn_processor,
        inputs=[model_input],
        outputs=[
            ProcessingOutput(output_name="result", source="/opt/ml/processing/result")
        ],
        code=os.path.join(source_dir, "postprocess.py"),
        property_files=[test_report],
        cache_config=cache_config
    )

#     model_package_group_name = "ts-prediction-informer"

    # Register model step that will be conditionally executed
    model_metrics = ModelMetrics(
        model_statistics=MetricsSource(
            s3_uri="{}/test_report.json".format(
                postprocessing_step.arguments["ProcessingOutputConfig"]["Outputs"][0]["S3Output"]["S3Uri"],
            ),
            content_type="application/json",
        )
    )    


    register_step = RegisterModel(
        name="InformerRegisterModel",
        estimator=estimator,
        model_data=training_step.properties.ModelArtifacts.S3ModelArtifacts,
        content_types=["text/csv"],
        response_types=["text/csv"],
        inference_instances=["ml.m5.xlarge"],
        transform_instances=["ml.m5.xlarge"],
        model_package_group_name=model_package_group_name,
        approval_status=model_approval_status,
        model_metrics=model_metrics,
    )


    # Condition step for evaluating model quality and branching execution
    cond_lte = ConditionLessThanOrEqualTo(  # You can change the condition here
        left=JsonGet(
            step=postprocessing_step,
            property_file=test_report,
            json_path="regression_metrics.mse.value",  # This should follow the structure of your report_dict defined in the postprocess.py file.
        ),
        right=1.0,  # You can change the threshold here
    )
    cond_step = ConditionStep(
        name="TestMSECond",
        conditions=[cond_lte],
        if_steps=[register_step],
        else_steps=[],
    )


    pipeline = Pipeline(
        name=pipeline_name,
        parameters=[train_instance_param, train_count_param, source_dir, model_approval_status],
        steps=[
            training_step,
            postprocessing_step,
            cond_step
        ],
    )
    return pipeline
