######################################################################################################
# To install this package:
# pip install --extra-index-url=https://nexus.smeir.io/repository/pypi-hosted/simple hateSpeechDetect
######################################################################################################

mkdir -p hateSpeechDetect/pretrained
rsync -rL checkpoint/ hateSpeechDetect/pretrained/

python3 setup.py sdist bdist_wheel
rm -r build hateSpeechDetect.egg-info
twine upload --repository-url=https://nexus.smeir.io/repository/pypi-hosted/ dist/* --username=dwshin
rm -r dist
