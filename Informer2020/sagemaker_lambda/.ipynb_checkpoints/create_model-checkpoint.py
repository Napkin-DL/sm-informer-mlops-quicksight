import json
import boto3
import os
from time import strftime

def lambda_handler(event, context):
    """
    모델 레지스트리에서 최신 버전의 모델 승인 상태를 변경하는 람다 함수.
    """
    
    try:
        sm_client = boto3.client("sagemaker")

        ##############################################
        # 람다 함수는 Event Bridge의 패턴 정보를 event 개체를 통해서 받습니다.
        ##############################################   
        print(f"event : {event}")
        model_package_arn = event['detail']["ModelPackageArn"]
        model_package_group_name = event['detail']["ModelPackageGroupName"]
        print("model_package_arn: ", model_package_arn)      
        print("model_package_group_name: ", model_package_group_name)
        
        # 패키지에 대한 모델 가져옵니다.
        container_list = [
            {
                "ModelPackageName": model_package_arn, 
                "Environment": {"SAGEMAKER_PROGRAM": "predictor.py"}
            }
        ]

        try:
            sm_client.delete_model(ModelName=model_package_group_name)
        except:
            pass

        create_model_response = sm_client.create_model(
            ModelName=model_package_group_name,
            ExecutionRoleArn=os.environ['role'],
            Containers=container_list
        )
        print("Model arn : {}".format(create_model_response["ModelArn"]))
        
        default_bucket = os.environ['default_bucket']
        
        ## Transformjob
        response = sm_client.create_transform_job(
          TransformJobName=model_package_group_name+"-"+strftime("%m%d-%H%M%s"),
          ModelName=model_package_group_name,
          MaxConcurrentTransforms=1,
          TransformInput={
              'DataSource': {
                  'S3DataSource': {
                      'S3DataType': 'S3Prefix',
                      'S3Uri': f's3://{default_bucket}/ETDataset/ETT-small/ETTh1.csv'
                  }
              },
              'ContentType' : 'text/csv',
              'SplitType': 'Line'
          },
          TransformOutput={
              'S3OutputPath': f"s3://{default_bucket}/batch_result",
              'AssembleWith': 'Line',
          },
          TransformResources={
              'InstanceType': 'ml.m5.xlarge',
              'InstanceCount': 1
          },
          Environment={
              'default_bucket': default_bucket
          },
        )

        return_msg = f"Success"
        
        ##############################################        
        # 람다 함수의 리턴 정보를 구성하고 리턴 합니다.
        ##############################################        

        return {
            "statusCode": 200,
            "body": json.dumps(return_msg),
            "other_key": "example_value",
        }

    except BaseException as error:
        return_msg = f"There is no model_package_group_name {model_package_group_name}"                
        error_msg = f"An exception occurred: {error}"
        print(error_msg)    
        return {
            "statusCode": 500,
            "body": json.dumps(return_msg),
            "other_key": "example_value",
        }        
        

