find . -name \*.so -delete
find . -name \*.pyc -delete
find . -name \*.c -delete
find . -name \*~ -delete
find . -name __pycache__ -delete
find . -name libipopt* -delete
find . -name libcoinmumps* -delete
find . -name libClp* -delete
find . -name libCbc* -delete
rm -rf OPTALG.egg-info
rm -rf build
rm -rf dist
rm -rf lib/ipopt
rm -rf lib/clp
rm -rf lib/cbc
rm -f lib/Ipopt*
rm -f lib/Clp*
rm -f lib/Cbc*
