# openmpi 2.1.1-2
setup:
	rm -rf venv
	virtualenv --python=python3 venv
	source venv/bin/activate &&\
		pip install --force -r requirements.txt &&\
		python setup.py build_ext --inplace
