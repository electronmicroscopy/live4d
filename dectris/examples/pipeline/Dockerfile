# The receiver pipeline (app.py + its operators in operators.py) on the imx-receiver base image
# (dectris/tools/build_base_image.sh); everything else is imported from the installed dectris package.
#
#   docker build -t imx-pipeline-example examples/pipeline
#   examples/pipeline/run.sh [--config <file>]   # default: the pipeline.yaml in the image
ARG BASE_IMAGE=imx-receiver-base:0.1.0
FROM ${BASE_IMAGE}

WORKDIR /app
COPY app.py operators.py settings.py pipeline.yaml ./

# Arguments of `docker run <image> ...` go straight to app.py.
ENTRYPOINT ["python3", "app.py"]
CMD ["--help"]
