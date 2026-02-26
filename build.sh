rm -rf build &&
pip uninstall flood -y && 
rm -rf /opt/conda/lib/python3.10/site-packages/flood* &&
rm -rf flood_cuda* &&
MAX_JOBS=16 python setup.py develop &&
echo "finish python setup.py develop!" 
# python setup.py bdist_wheel &&
# echo "finish python setup.py bdist_wheel!"