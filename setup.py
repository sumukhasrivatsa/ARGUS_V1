from setuptools import setup

package_name = 'argus_v1'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Sumukha Srivatsa',
    maintainer_email='sumukha@gatech.edu',
    description='ARGUS: Continuous LLM-Grounded Perception for Reactive Robot Manipulation',
    license='MIT',
    entry_points={
        'console_scripts': [
            'visual_node = argus_v1.VisualBlock:main',
        ],
    },
)