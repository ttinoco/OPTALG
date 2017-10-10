find . -name \*.so -delete
find . -name \*.pyc -delete
find . -name \*.c -delete
find . -name libipopt* -delete
find . -name libcoinmumps* -delete
find . -name libClp* -delete
rm -rf OPTALG.egg-info
rm -rf build
rm -rf dist
rm -rf lib/ipopt
rm -rf lib/clp
rm lib/Ipopt*
rm lib/Clp*
