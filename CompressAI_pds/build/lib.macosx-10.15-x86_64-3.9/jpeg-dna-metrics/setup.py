"""Setting up package for pip"""
from setuptools import setup, find_packages
with open("README.md", "r", encoding="UTF-8") as fh:
    long_description = fh.read()

setup(name="JPEG DNA Metrics definition",
      version="0.0b1",
      author="Xavier Pic, Eva Gil San Antonio, Melpomeni Dimopoulou, Marc Antonini",
      author_email="xpic@i3s.unice.fr, gilsanan@i3s.unice.Fr, dimopoulou@i3s.unice.fr, am@i3s.unice.fr",
      description="Library gathering metrics \
        for compressing and encoding images into DNA",
      long_description=long_description,
      long_description_content_type="test/markdown",
      packages=find_packages(),
      python_requires=">=3.8",
      install_requires=["scikit-image", "argparse", "opencv-python"],
      # packages=['compression', 'quality'],
      package_dir={'jdna_compression':'compression', 'jdna_quality': 'quality'},
      entry_points="""
      [console_scripts]
      jdna_psnr = quality.psnr:main
      jdna_rate = compression.rates:main
      """)
