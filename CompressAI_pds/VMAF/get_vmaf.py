import subprocess
import json
import pandas as pd

JPEG_SIZES = [
    '1192x832', 
    '853x945', 
    '945x840', 
    '2000x2496', 
    '560x888',
    '2048x1536', 
    '1600x1200', 
    '1430x1834', 
    '2048x1536', 
    '2592x1946'  
]


for data_set in ['jpeg_dna']: 
	for model_name in ['learningbased']:
	#for model_name in ['anchor1', 'anchor3', 'benchmarkcodec', 'BCtranscoder', 'learningbased']:

		original_img_path = '/Users/noemiemonnier/Documents/COURS/pds/CompressAI_pds/examples/assets/' + data_set + '/'
		decoded_img_path = '/Users/noemiemonnier/Documents/COURS/pds/CompressAI_pds/examples/assets/' + data_set + '/' + model_name + '/'

		if data_set == 'kodak': 
		    number_of_images = 24
		else: 
		    number_of_images = 10

		if data_set == 'jpeg_dna' and model_name == 'learningbased': 
			decoded_img_path += 'nopadding/'

		# Create a dictionary to store the vmaf values
		vmaf_values = {'Image': [], 'Quality': [], 'VMAF': []}

		for i in range(1, number_of_images + 1): 
		    if data_set == 'kodak': 
		        original_img_name = 'kodim{:02d}.png'.format(i)
		    else: 
		        original_img_name = f"{str(i).zfill(5)}" + '_' + JPEG_SIZES[i-1] + '.png'

		    for q in range(1, 9): 
		        if data_set == 'kodak': 
		            decoded_img_name = '{}kodim{:02d}.png'.format(q, i)
		        else: 
		            decoded_img_name = f"JPEG-1_{str(i).zfill(5)}" + '_' + JPEG_SIZES[i-1] + '_' + str(q) + '_decoded.png'
		        output = subprocess.run(["ffmpeg-quality-metrics", decoded_img_path + decoded_img_name, original_img_path + original_img_name, "--metrics", "vmaf"], capture_output=True, text=True)

		        output_data = json.loads(output.stdout)
		        vmaf_value = output_data['vmaf'][0]['vmaf']
		        print(vmaf_value)
		        
		        # Append the vmaf value to the dictionary
		        vmaf_values['Image'].append(i)
		        vmaf_values['Quality'].append(q)
		        vmaf_values['VMAF'].append(vmaf_value)

		# Create a pandas DataFrame from the dictionary
		df = pd.DataFrame(vmaf_values)

		# Save the DataFrame to an Excel file
		output_file = f"{model_name}_{data_set}_vmaf.xlsx"
		df.to_excel(output_file, index=False)

		print(f"VMAF values saved to {output_file}")

