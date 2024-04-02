from ygong.mosaic import submit, _set_up_environment
from ygong.mosaic import ScalingConfig
from ygong.mosaic import TrainingConfig
from ygong.mosaic.wsfs import WSFSIntegration

import json
import sys

def main():
    import base64
    data = {
        'workspace_url': 'https://e2-dogfood.staging.cloud.databricks.com/',
        'token': 'dapid5af61ff89674be90c3e86ae9fc10c2e'
    }

    data = json.dumps(data)
    content = base64.b64encode(data.encode('utf-8')).decode('utf-8')
    _set_up_environment(content)

    import argparse
    parser = argparse.ArgumentParser(description='Submit a run to mosaic')
    parser.add_argument('--parameters_json', type=str, help='parameters_json',
                        default = """
                            {
                                "name": "custom-train-demo", 
                                "seed": 42, 
                                "device_train_microbatch_size": 8,
                                "loggers": {
                                    "mlflow": {
                                        "tracking_uri": "databricks",
                                        "synchronous": false,
                                        "log_system_metrics": true
                                    }
                                }
                            }
                        """
                        )
    parser.add_argument('--custom_code_repo_dir', type=str, help='custom_code_repo_dir', default="/Workspace/Users/yu.gong@databricks.com/.ide/custom-train-demo-4bf5c137")
    parser.add_argument('--pool_name', type=str, help='pool_name', default="staging-aws-us-east-1-mlserv1-gentrain1")
    parser.add_argument('--priority', type=str, help='priority', default="high")
    parser.add_argument('--preemptible', type=bool, help='preemptible', default=False)
    parser.add_argument('--retry_on_system_failure', type=bool, help='retry_on_system_failure', default=False)
    parser.add_argument('--entry_point', type=str, help='entry_point', default="./src/train.py")
    args, _ = parser.parse_known_args()


    parameters_json = args.parameters_json
    custom_code_repo_dir = args.custom_code_repo_dir
    pool_name = args.pool_name
    priority = args.priority
    preemptible = args.preemptible
    retry_on_system_failure = args.retry_on_system_failure
    entry_point = args.entry_point

    scalingConfig = ScalingConfig(
        gpusNum=8,
        poolName=pool_name,
        priority = priority,
        preemptible= preemptible,
        retry_on_system_failure= retry_on_system_failure
    )

    config = TrainingConfig(
        name="custom-train-demo",
        entry_point=f'{custom_code_repo_dir}{entry_point}', 
        parameters=json.loads(parameters_json)
    )

    # DEMO NOTE:
    # This is temporary hack to mock the behavior of mounting workspace filesystem to the remote training nodes. Once the workspace filesystem fusion is integrated with netphos and dblet. We will get this for free.
    wsfs = WSFSIntegration(
        wsfs_path=custom_code_repo_dir)

    submit(config, scalingConfig, wait_job_to_finish=True, debug=False, wsfs=wsfs)

if __name__ == "__main__":
    sys.exit(main())
