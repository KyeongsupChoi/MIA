from setuptools import find_packages, setup

setup(
    name='mia',
    packages=find_packages(),
    version='0.1.0',
    description='Chest X-ray pneumonia classifier using ResNet50 transfer learning',
    author='Kyeongsup Choi',
    license='MIT',
    python_requires='>=3.10',
    entry_points={
        'console_scripts': [
            'mia-train=src.models.train_Carmine400:main',
            'mia-predict=src.models.predict_model:main',
            'mia-gradcam=src.visualization.gradcam:main',
        ],
    },
)
