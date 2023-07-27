# Examples

All the plots can be computed from the model_testing.ipynb file. The results are stored in 'CompressAI_pds/examples/plots'

The possible data sets are : 'kodak', 'jpeg_dna'

The possible metrics are : 'PSNR',
    'PSNR-YUV',
    'MS-SSIM',
    'IW-SSIM',
    'NLPD',
    'VIF',
    'FSIM',
    'VMAF',
    'Homopolymer density',
    'Homopolymer average length',
    'Homopolymer frequency'

  The possible models are : 'benchmarkcodec',
    'anchor1',
    'anchor3',
    'learningbased',
    'BCtranscoder'

  - CG contents
plot_cg_contents(model_name, data_set = 'jpeg_dna'):
    generates plot for given model and data set
plot_all_cg_contents() :
    generates plots for all models on all data sets

  - Homopolymers
plot_homopolymers(data_set, model_name, img_number):
    generates box plot for one image with given model

plot_model_homopolymers(data_set, model_name)
    generates box plot for all the images of the given data set for one model

plot_metric(data_set, metric_name, img_number) with metric_name = 'Homopolymer frequency' or 'Homopolymer length'
    generates plot for the required data for one image for all models at different rates

plot_avg_metric(metric_name, data_set = 'jpeg_dna')  with metric_name = 'Homopolymer frequency' or 'Homopolymer length'
    generates plot for the required data averaged over the images of the data set for all models at different rates


    - Quality metrics
plot_metric(data_set, metric_name, img_number)
    generates plot for the required metric for one image for all models at different rates

plot_avg_metric(metric_name, data_set = 'jpeg_dna')  with metric_name = 'Homopolymer frequency' or 'Homopolymer length'
    generates plot for the required metric averaged over the images of the data set for all models at different rates

    ## NOTE : This fucntion computes all the metrics, except for VMAF that was computed once and stored in a table. If needed, to update those values, use the script 'CompressAI_pds/VMAF/get_vmaf.py'


    - Subjective inspection
This is done directly in the notebook, and saves the resulting images & plots in 'examples/assets/subjective eval'



The learning based assets can be computed from the file GetLearningBasedImages.ipynb. The compute_x_hat function should be updated in the file directly with the new one, and the files will be stored where they can be used for the plots.

## Notebooks

To run the jupyter notebooks:

* `pip install -U ipython jupyter ipywidgets matplotlib`
* `jupyter notebook`
