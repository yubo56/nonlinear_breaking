# openmpi 2.1.1-2
setup:
	. venv/bin/activate &&\
		pip install -r requirements.txt &&\
		python setup.py build_ext --inplace

venv: 
	virtualenv --python=python3 venv

clean:
	rm -rf venv
	find . -iname *.c -delete && find . -iname *.so -delete
