from distutils.core import setup, Extension

gyft_extension = Extension(
    'gyftss',
    include_dirs=['src'],
    libraries=[
        'opencv_core', 'opencv_imgproc', 'opencv_photo', 'opencv_imgcodecs',
        'lept', 'tesseract', 'python3'
    ],
    extra_compile_args=['-std=c++11'],
    language='c++',
    sources=['src/gyftss.cpp'])

setup(
    name='gyftss',
    version='1.0',
    description='Get IITKGP timetable from screenshot',
    ext_modules=[gyft_extension])
