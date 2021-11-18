#plot directly
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# arr = np.load('/Users/juexinwang/workspace/scGNNsp/model1_1/coords_array.npy')
# df = pd.read_csv('/Users/juexinwang/workspace/scGNNsp/outputdir/sim_results.txt')
arr = np.load('/Users/wangjue/workspace/scGNNsp/model1_1/coords_array.npy')
df = pd.read_csv('/Users/wangjue/workspace/scGNNsp/outputdirS-model1_1/model1_1_8_euclidean_NA_dummy_add_0.5_results.txt')
# df = pd.read_csv('/Users/juexinwang/Documents/scGNNsp_results/151673_cosine_dummy_add_0.5_results_2_all.txt')
# df = pd.read_csv('/Users/wangjue/Documents/results_scGNNsp/cpm_0.3/151673_cpm_8_cityblock_NA_geom_lowf_add_0.5_results.txt')
# df = pd.read_csv('/home/wangjue/workspace/scGNNsp/outputdir-151673/151673_correlation_dummy_add_0.1_results_4.txt')
# # df = pd.read_csv('/home/wangjue/workspace/scGNNsp/outputdirscgnn/151673_noregu_0.9_1.0_0.0_results_0.txt')
# sns.lmplot('population', 'Area', data=df, hue='continent', fit_reg=False)
# sns.lmplot(data=df, fit_reg=False)
color_labels = df['Celltype'].unique()
print(color_labels)
# List of colors in the color palettes
rgb_values = sns.color_palette("Set2", len(color_labels))
# Map continents to the colors
color_map = dict(zip(color_labels, rgb_values))
# Finally use the mapped values
plt.scatter(arr[:,0], arr[:,1], c=df['Celltype'].map(color_map), s=10)
plt.show()
plt.savefig('tt.png')


##########

# #plot
# t=np.load('defaultPE.npy')

# tmp=[]
# i=0
# for j in range(10):
#     tmp.append(t[i,j]+t[i,j+10])

# plt.plot(tmp,'y.')

# tmp=[]
# i=2
# for j in range(10):
#     tmp.append(t[i,j]+t[i,j+10])

# plt.plot(tmp,'b.')

# tmp=[]
# i=20
# for j in range(10):
#     tmp.append(t[i,j]+t[i,j+10])

# plt.plot(tmp,'r.')

# plt.show()



# ##########
# tmp=[]
# i=0
# for j in range(10):
#     tmp.append(t[i,j])

# plt.plot(tmp,'y.')

# tmp1=[]
# i=2
# for j in range(10):
#     tmp1.append(t[i,j])

# plt.plot(tmp1,'b.')

# tmp2=[]
# i=10
# for j in range(10):
#     tmp2.append(t[i,j])

# plt.plot(tmp2,'r.')

# plt.show()