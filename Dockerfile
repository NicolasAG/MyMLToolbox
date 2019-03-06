# start from pytorch
FROM pytorch/pytorch

# create conda environment
ADD conda-env.txt /env/conda-env.txt
RUN conda create --name py37myMLtoolbox --file /env/conda-env.txt

# conda activate & source activate doesn't work,
# so add the environment path to the PATH
ENV PATH /opt/conda/envs/py37myMLtoolbox/bin:$PATH


# copy current repository
COPY . /code/MyMLToolbox

# install dependencies
RUN pip install -r /code/MyMLToolbox/requirements.txt

# clone the models
RUN cd /code/MyMLToolbox/external && ./setup.sh
