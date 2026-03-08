# Use Ubuntu 24.04 as base image
FROM ubuntu:24.04

# Set non-interactive mode to avoid prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install required dependencies
RUN apt-get update && apt-get install -y \
    make \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    curl \
    git \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev

# Switch to 'ubuntu' user for pyenv installation
USER ubuntu
WORKDIR /home/ubuntu

# Set shell
SHELL ["/bin/bash", "-c"]

# Set environment variables for pyenv
ENV PYENV_ROOT=/home/ubuntu/.pyenv
ENV PATH=$PYENV_ROOT/bin:$PATH

# Install pyenv
RUN curl -fsSL https://pyenv.run | bash 

RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bashrc
RUN echo '[[ -d $PYENV_ROOT/bin ]] ' >> ~/.bashrc
RUN echo 'export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bashrc
RUN echo 'eval "$(pyenv init - bash)"' >> ~/.bashrc

#RUN echo 'export PYENV_ROOT="$HOME/.pyenv"' >> ~/.bash_profile 
#RUN echo '[[ -d $PYENV_ROOT/bin ]] && export PATH="$PYENV_ROOT/bin:$PATH"' >> ~/.bash_profile 
#RUN echo 'eval "$(pyenv init - bash)"' >> ~/.bash_profile 
#RUN bash -c "source ~/.bashrc && source ~/.bash_profile"

# Install Python 3.13.6 using pyenv
RUN pyenv install 3.13.6 
RUN $HOME/.pyenv/versions/3.13.6/bin/python -m venv python3.13venv
ENV PATH="/home/ubuntu/python3.13venv/bin:$PATH"

# Verify installation
RUN python --version

# Set working directory
WORKDIR /home/ubuntu

# Install Comfy-UI
RUN pip install --force-reinstall torch diffusers gguf accelerate transformers torchvision llama-cpp-python --extra-index-url https://download.pytorch.org/whl/cu130
RUN pip install comfy-cli
RUN comfy --skip-prompt install --nvidia

# Run Comfy-UI
EXPOSE 8080
CMD ["comfy", "launch", "--", "--listen", "0.0.0.0", "--port", "8080"]