# fluffy-pancake
stable diffusion, image inference python scripts

# Python installation
Specific to the MSI EdgeXpert variant of the Nvidia DGX Spark

Rough notes below.

```
curl -fsSL https://pyenv.run | bash 
```
 
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc 

echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc 

echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc 
```
 
```
echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile 

echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile 

echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile 
```
 
```
exec "$SHELL" 
```
 
```
sudo apt update; sudo apt install make build-essential libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev curl git libncursesw5-dev xz-utils tk-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev 
```
 
```
pyenv install -v 3.12.9 
```
 
```
jack@io:~/repos/fluffy-pancake$ /home/jack/.pyenv/versions/3.12.9/bin/python -m venv python3.12venv 

jack@io:~/repos/fluffy-pancake$ source python3.12venv/bin/activate 

# have to re-download and compile for ARM processor. 
(python3.12venv) jack@io:~/repos/fluffy-pancake$ pip install --force-reinstall torch diffusers gguf accelerate transformers torchvision llama-cpp-python --extra-index-url https://download.pytorch.org/whl/cu130 
```
 
Only needed if an older pytorch cuda index url was initially used.
```
https://forums.developer.nvidia.com/t/effective-pytorch-and-cuda/348230/13 

(python3.12venv) jack@io:~/repos/fluffy-pancake$ pip install –force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130 
```
