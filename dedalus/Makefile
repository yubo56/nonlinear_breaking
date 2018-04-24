# openmpi 2.1.1-2
setup: clean
	virtualenv --python=python3 venv
	source venv/bin/activate &&\
		pip install --force -r requirements.txt &&\
		python setup.py build_ext --inplace
clean:
	rm -rf venv
	find . -iname *.c -delete && find . -iname *.so -delete
