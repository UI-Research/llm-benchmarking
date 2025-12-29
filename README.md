# llm-benchmarking

## To get started

1. Clone this repository to your local machine using.

2. Create and activate a virtual environment.
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    ```

3. Install the required packages.
    ```bash
    pip install -r requirements.txt
    ```

4. Install `poppler utils` for Unstructured data partitioning:
    ```bash
    sudo apt update

    sudo apt-get update

    sudo apt-get install -y poppler-utils

    sudo apt-get install tesseract-ocr
    ```

5. To access AWS Bedrock models, set up your AWS credentials. You can do this by configuring the AWS CLI or setting environment variables. For more information, refer to the [AWS documentation](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html).


## Archive of Previous Versions

## License Agreements

To access the Llama 4 family of models through Hugging Face, 
1. Create an account on [Hugging Face](https://huggingface.co/join).
2. Accept the Llama 4 license agreement displayed on [this page] (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
3. Await the status of your request to be updated to "Accepted" [here] (https://huggingface.co/settings/gated-repos). 
4. Install huggingface-cli using `pip install -U "huggingface_hub[cli]"`.
5. Log in to your Hugging Face account** using `huggingface-cli login`. You will be prompted to provide a token which can be generated [here] (https://huggingface.co/settings/tokens). Read more on the CLI login [here] (https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-whoami).
6. Check if your token is active using `huggingface-cli whoami`.

**Note: Login to Hugging Face outside of the virtual environment otherwise you will be asked to provide credentials every time you work within the virtual environment. 