# llm-benchmarking

## License Agreements

To access the Llama 4 family of models through Hugging Face, 
1. Create an account on [Hugging Face](https://huggingface.co/join).
2. Accept the Llama 4 license agreement displayed on [this page] (https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct).
3. Await the status of your request to be updated to "Accepted" [here] (https://huggingface.co/settings/gated-repos). 
4. Install huggingface-cli using `pip install -U "huggingface_hub[cli]"`.
5. Log in to your Hugging Face account** using `huggingface-cli login`. You will be prompted to provide a token which can be generated [here] (https://huggingface.co/settings/tokens). Read more on the CLI login [here] (https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#huggingface-cli-whoami).
6. Check if your token is active using `huggingface-cli whoami`.

**Note: Login to Hugging Face outside of the virtual environment otherwise you will be asked to provide credentials every time you work within the virtual environment. 