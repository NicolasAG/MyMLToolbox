FROM pytorch/pytorch

ADD conda-env.txt /env/conda-env.txt
RUN conda create --name py37myMLtoolbox --file /env/conda-env.txt

# RUN echo "source activate py37myMLtoolbox" > ~/.bashrc

