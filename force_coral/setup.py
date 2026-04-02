from setuptools import find_packages, setup

setup(
    name="force_coral",
    version="0.1.0",
    description="Custom robotics experiments built on LIBERO + FoundationPose",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.24,<2.0",
        "scipy>=1.10,<1.14",
        "matplotlib>=3.7",
        "opencv-python>=4.8",
        "Pillow>=10.0",
        "imageio[ffmpeg]>=2.34",
        "python-dotenv>=1.0",
        "easydict==1.9",
        "cloudpickle==2.1.0",
        "gym==0.25.2",
    ],
    extras_require={
        "sim": [
            "mujoco>=3.1",
            "robosuite==1.4.0",
            "bddl==1.0.1",
            "transformations>=2024.6.1",
            "trimesh>=4.0",
        ],
        "llm": [
            "openai>=1.0",
        ],
        "dev": [
            "pytest>=8.0",
            "pytest-cov>=5.0",
        ],
    },
)
