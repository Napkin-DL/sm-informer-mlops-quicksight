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
import json

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
from sagemaker.processing import FrameworkProcessor
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

###############################################################
###############################################################
### 2번 노트북 - '4. Experments 관리' 부터 복사해서 붙여넣기를 합니다. ###
###############################################################
###############################################################

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

    ### 4. Experiments 관리    
    def create_experiment(experiment_name):
        try:
            sm_experiment = Experiment.load(experiment_name)
        except:
            sm_experiment = Experiment.create(experiment_name=experiment_name,
                                              tags=[{'Key': 'modelname', 'Value': 'informer'}])


    def create_trial(experiment_name, i_type, i_cnt, spot=False):
        create_date = strftime("%m%d-%H%M%s")
        algo = 'informer'

        spot = 's' if spot else 'd'
        i_type = i_type[3:9].replace('.','-')

        trial = "-".join([i_type,str(i_cnt),algo, spot])

        sm_trial = Trial.create(trial_name=f'{experiment_name}-{trial}-{create_date}',
                                experiment_name=experiment_name)

        job_name = f'{sm_trial.trial_name}'
        return job_name

    
    ### 5. 실험 설정
    code_location = f's3://{default_bucket}/poc_informer/sm_codes'
    output_path = f's3://{default_bucket}/poc_informer/output'         
        
        
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
     
    experiment_name = 'informer-poc-exp1'                                                          ### <== 1. Experiments 이름 수정
    distribution = None
    do_spot_training = True
    max_wait = None
    max_run = 1*30*60
        
    instance_type="ml.m5.xlarge"                                                                   
    instance_count=1        
    
    
    ### 6. Pipeline parameters, checkpoints와 데이터 위치 설정
    #### 6-1. Pipeline parameters
    train_instance_param = ParameterString(
        name="TrainingInstance",
        default_value="ml.c5.4xlarge",                                                             ### <== 2. Instance 타입, 개수 수정
    )

    train_count_param = ParameterInteger(
        name="TrainingInstanceCount",
        default_value=1
    )

    model_approval_status = ParameterString(
        name="ModelApprovalStatus", default_value="PendingManualApproval"
    )
    
    #### 6-2. checkpoints와 데이터 위치 설정
    image_uri = None
    train_job_name = 'informer'

    if do_spot_training:
        max_wait = max_run

    print("train_job_name : {} \ntrain_instance_type : {} \ntrain_instance_count : {} \nimage_uri : {} \ndistribution : {}".format(train_job_name, train_instance_param.default_value, train_count_param.default_value, image_uri, distribution))    

    
    prefix = 'ETDataset'
    inputs = f's3://{default_bucket}/dataset/{prefix}'

    source_dir = 'Informer2020'                                                                    ### <== 3. git repository내의 소스 코드 위치 
    checkpoint_s3_uri = f's3://{default_bucket}/poc_informer/checkpoints'      
    
    
    #### 6-3. Git 설정 (Secret Manager 활용)
    def get_secret(secret_name):
        secret = {}
        # Create a Secrets Manager client
        session = boto3.session.Session()
        client = session.client(
            service_name='secretsmanager'
        )

        # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
        # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
        # We rethrow the exception by default.

        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )

        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            secret = json.loads(secret)
        else:
            print("secret is not defined. Checking the Secrets Manager")

        return secret        
        
    ### CodeCommit의 Credentials이 저장된 secret_name 사용이 필요합니다.
    sec_client = boto3.client('secretsmanager')
    secret_name = sec_client.list_secrets(SortOrder='desc')['SecretList'][0]['ARN']               ### <== 4. git credentials이 있는 secret manage 이름 수정
     
    secret=get_secret(secret_name)

    git_config = {'repo': 'https://git-codecommit.us-west-2.amazonaws.com/v1/repos/informer2020', ### <== 5. git repository 위치 수정
                  'branch': 'main',
                  'username': secret['username'],
                  'password': secret['password']}
        
        
    ### 7. 학습을 위한 Estimator 선언
    create_experiment(experiment_name)
    job_name = create_trial(experiment_name, instance_type, instance_count, spot=do_spot_training)


    estimator = PyTorch(
        entry_point='main_informer.py',
        source_dir=source_dir,
        git_config=git_config,
        role=role,
        sagemaker_session=sagemaker_session,
        framework_version='1.10',
        py_version='py38',
        instance_count=train_count_param,    ## Parameter 값으로 변경
        instance_type=train_instance_param,  ## Parameter 값으로 변경
        volume_size=256,
        code_location = code_location,
        output_path=output_path,
        hyperparameters=hyperparameters,
        distribution=distribution,
        metric_definitions=metric_definitions,
        max_run=max_run,
        checkpoint_s3_uri=checkpoint_s3_uri,
        use_spot_instances=do_spot_training,  # spot instance 활용
        max_wait=max_wait,
        base_job_name=f"training-{job_name}",
        disable_profiler=True,
        debugger_hook_config=False,
    )
    
    
    ### 8. Training 단계 선언    
    from sagemaker.workflow.steps import CacheConfig

    cache_config = CacheConfig(enable_caching=True, 
                               expire_after="7d")        
        
        
    training_step = TrainingStep(
        name="InformerTrain",
        estimator=estimator,
        inputs={
            "training": sagemaker.inputs.TrainingInput(
                s3_data=inputs
            )
        },
        cache_config=cache_config
    )        
        
    
    ### 9. Evaluation 단계 - output에서 압축풀어 test_report.json 가져오기
    framework_processor = FrameworkProcessor(
        PyTorch,
        framework_version="1.10",
        py_version='py38',
        role=role,
        instance_count=1,
        instance_type="ml.c4.xlarge",
        code_location=code_location,
        base_job_name=f"generatingreport-{job_name}",  # choose any name
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
        
    run_args = framework_processor.get_run_args(
        code="postprocess.py",
        source_dir="Informer2020",
        git_config=git_config,
        inputs=[model_input],
        outputs=[
            ProcessingOutput(output_name="result", source="/opt/ml/processing/result")
        ],
        job_name=f"process-step-{job_name}"
    )        
        

    postprocessing_step = ProcessingStep(
        name="PostProcessingforInformer",  # choose any name
        processor=framework_processor,
        inputs=run_args.inputs,
        outputs=run_args.outputs,
        code=run_args.code,
        property_files=[test_report],
        cache_config=cache_config
    )
        

    ### 10. Model 등록 단계
    model_package_group_name = 'mlops-test-informer-p-dkvarxpz6dj8'                                      ### <== 6. model package group 이름 수정
    
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

    
    ### 11. Condition 단계
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
    
    ### 12. Pipeline 수행
    pipeline = Pipeline(
        name="ts-prediction-informer-pipeline",
        parameters=[train_instance_param, train_count_param, model_approval_status],
        steps=[
            training_step,
            postprocessing_step,
            cond_step
        ],
    )

    return pipeline


