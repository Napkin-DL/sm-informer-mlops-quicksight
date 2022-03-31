# Copyright 2020 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#     http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.
"""Feature engineers the custom env"""

import os  
import tarfile


if __name__ == "__main__":
    
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--code-dir", type=str, required=True)
#     args = parser.parse_args()
    
    print(f"Uncompress a tar file")   
    base_dir = "/opt/ml/processing"
    
    print(f"result  : {os.listdir(base_dir)}")
    
    fname = base_dir + "/model/model.tar.gz"
    result_path = base_dir + "/result"
    
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall(path=base_dir + "/result")
        tar.close()
    
    for root, dir, file in os.walk(base_dir):
        print(root, dir, file)