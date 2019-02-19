# start from pytorch
FROM pytorch/pytorch

# create conda environment
ADD conda-env.txt /env/conda-env.txt
RUN conda create --name py37myMLtoolbox --file /env/conda-env.txt

# conda activate & source activate doesn't work,
# so add the environment path to the PATH
ENV PATH /opt/conda/envs/py37myMLtoolbox/bin:$PATH

# comet_ml is not supported by conda so add it manually
RUN pip install comet_ml

# install external libraries from MyMLtoolbox/external/
#ADD external/dowload_externals.sh /external/dowload_externals.sh
#RUN cd /external/ && ./dowload_externals.sh
