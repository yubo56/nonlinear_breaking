# openmpi 2.1.1-2
setup:
	rm -rf venv
	virtualenv3 venv --system-site-packages ||\
		virtualenv venv --system-site-packages
	source venv/bin/activate &&\
		pip install -r requirements.txt &&\
		python setup.py build_ext --inplace
