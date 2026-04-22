# get the base docker image
FROM public.ecr.aws/lambda/python:3.11
RUN yum install -y gcc gcc-c++
RUN pip install --upgrade pip

# copy requirements.txt
COPY requirements.txt ${LAMBDA_TASK_ROOT}

# install specified requirements
RUN pip install -r requirements.txt

# copy all the files in ./src
COPY src/* ${LAMBDA_TASK_ROOT}

# set the CMD to the handler
# in other words the lambda function runs main.handler when it is run
CMD [ "main.handler"]