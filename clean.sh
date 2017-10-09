find . -name \*.so -delete
find . -name \*.pyc -delete
find . -name \*.c -delete
find . -name libipopt* -delete
find . -name libcoinmumps* -delete
rm -rf OPTALG.egg-info
rm -rf build
rm -rf dist
rm -rf lib/ipopt
rm lib/Ipopt*
rm -rf lib/clp
rm lib/Clp*
