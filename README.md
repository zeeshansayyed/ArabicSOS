# ArabicSOS
Segmenter and Orthography Standardazier (SOS) for Classical Arabic (CA)

This is the beta version of the Arabic Segmenter for segmenting classical Arabic texts. It has currently been trained on a subset of Al-Manar corpus created by Dr. Emad Mohamed.

Related Paper: ![Arabic-SOS: Segmentation, Stemming, and Orthography Standardization for Classical and pre-Modern Standard Arabic](https://dl.acm.org/doi/abs/10.1145/3322905.3322927)

**Disclaimer**:
This package is still in the early development stages, hence the documentation is sparse. While it has been tested in standard use-cases, there might be a few bugs in the code. Please make sure you have a backup of your data before you use the package. We will greatly appreciate any feedback at the email addresses listed in the controbutors section.

**Requirements**:
1. Python 3.x
2. pandas >= 0.23.4 (pip install pandas)
3. catboost >= 0.11.2 (pip install catboost)

**Model Files**:
    Please download and install the model files in the model folder using the following command:
        `wget -v -O catboost_1.model -L https://iu.box.com/shared/static/mcu4frnipinfw7ery0wetrax7u7zzsxp.model`
    
**Usage**:
    `python segmenter.py input_file_path -o _output_file_path`

- Providing the input_file path is mandatory
- If you do not provide the output_file, it will be created in the same directory as that of the input file. The name of the input file will be appended by  ".segmented"

Note: The package assumes that every line in the file contains a single sentence.

    
**Example**:
    There is a file named P105.txt in the sample folder. It contains raw arabic text. We can segment it as follows:
    `python segmenter.py sample/P105.txt -o sample/my_segmented_file.txt`
Or simply  
    `python segmenter.py sample/P105.txt`
which will result in the creation of P105.txt.segmented file in the sample folder


**Contributors**:
1. Zeeshan Ali Sayyed (zasayyed@iu.edu)
2. Emad Mohamed (emohamed@umail.iu.edu)


**Acknowledgment**:
“This project was made possible by NPRP grant NPRP10-0115-170163 from the Qatar National Research Fund (a member of Qatar Foundation). The findings achieved herein are solely the responsibility of the authors”.

 
    
