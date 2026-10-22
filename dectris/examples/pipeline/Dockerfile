# The receiver pipeline (pipeline.py + its operators in operators.py) on the imx-receiver base image
# (dectris/tools/build_base_image.sh); everything else is imported from the installed dectris package.
#
#   docker build -t imx-pipeline-example examples/pipeline
#   examples/pipeline/run.sh 10.42.41.10:31003 --local-addr 10.42.41.11 --seconds 0
ARG BASE_IMAGE=imx-receiver-base:0.1.0
FROM ${BASE_IMAGE}

WORKDIR /app
COPY pipeline.py operators.py ./

# Arguments of `docker run <image> ...` go straight to pipeline.py.
ENTRYPOINT ["python3", "pipeline.py"]
CMD ["--help"]
