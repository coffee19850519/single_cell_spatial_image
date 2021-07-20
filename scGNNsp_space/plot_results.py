#plot directly
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plotResults(dataName, dirName='S',para='_8_euclidean_Grid_dummy_add_0.5'):
    arr = np.load('/storage/htc/joshilab/wangjue/scGNNsp/'+dataName+'/coords_array.npy')
    df = pd.read_csv('/storage/htc/joshilab/wangjue/scGNNsp/outputdir'+dirName+'-'+dataName+'_0.3/'+dataName+para+'_results.txt')
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
    plt.savefig(str(dataName)+'-'+dirName+'-'+para+'.png')
    plt.close()

# Basic Spatial
print('Basic Spatial')
plotResults('151507_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151508_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151509_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151510_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151669_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151670_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151671_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151672_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151673_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151674_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151675_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151676_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('18-64_cpm',  para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('2-5_cpm',    para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('2-8_cpm',    para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('T4857_cpm',  para='_8_euclidean_Grid_dummy_add_0.5')

# Best Spatial
print('Best Spatial') 
plotResults('151507_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151508_cpm', para='_40_euclidean_GridEx4_dummy_add_0.5_intersect')
plotResults('151509_cpm', para='_40_euclidean_GridEx4_dummy_add_0.5_intersect')
plotResults('151510_cpm', para='_64_euclidean_GridEx7_dummy_add_0.5_intersect')
plotResults('151669_cpm', para='_104_euclidean_GridEx12_dummy_add_0.5_intersect')
plotResults('151670_cpm', para='_32_euclidean_GridEx3_dummy_add_0.5_intersect')
plotResults('151671_cpm', para='_96_euclidean_GridEx11_dummy_add_0.5_intersect')
plotResults('151672_cpm', para='_48_euclidean_GridEx5_dummy_add_0.5_intersect')
plotResults('151673_cpm', para='_16_euclidean_GridEx_dummy_add_0.5_intersect')
plotResults('151674_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151675_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('151676_cpm', para='_8_euclidean_Grid_dummy_add_0.5')
plotResults('18-64_cpm',  para='_32_euclidean_GridEx3_dummy_add_0.5_intersect')
plotResults('2-5_cpm',    para='_32_euclidean_GridEx3_dummy_add_0.5_intersect')
plotResults('2-8_cpm',    para='_16_euclidean_GridEx_dummy_add_0.5_intersect')
plotResults('T4857_cpm',  para='_16_euclidean_GridEx_dummy_add_0.5_intersect')

# Best Hybrid
print('Best Hybrid')  
plotResults('151507_cpm', dirName='H', para='_200_euclidean_NA_dummy_add_0.5_intersect_16_GridEx')
plotResults('151508_cpm', dirName='H', para='_2000_euclidean_NA_dummy_add_0.5_intersect_160_GridEx19')
plotResults('151509_cpm', dirName='H', para='_500_euclidean_NA_dummy_add_0.5_intersect_200_GridEx24')
plotResults('151510_cpm', dirName='H', para='_50_euclidean_NA_dummy_add_0.5_intersect_80_GridEx9')
plotResults('151669_cpm', dirName='H', para='_1000_euclidean_NA_dummy_add_0.5_intersect_200_GridEx24')
plotResults('151670_cpm', dirName='H', para='_1000_euclidean_NA_dummy_add_0.5_intersect_160_GridEx19')
plotResults('151671_cpm', dirName='H', para='_100_euclidean_NA_dummy_add_0.5_intersect_80_GridEx9')
plotResults('151672_cpm', dirName='H', para='_2000_euclidean_NA_dummy_add_0.5_intersect_80_GridEx9')
plotResults('151673_cpm', dirName='H', para='_2000_euclidean_NA_dummy_add_0.5_intersect_40_GridEx4')
plotResults('151674_cpm', dirName='H', para='_10_euclidean_NA_dummy_add_0.5_intersect_16_GridEx')
plotResults('151675_cpm', dirName='H', para='_10_euclidean_NA_dummy_add_0.5_intersect_40_GridEx4')
plotResults('151676_cpm', dirName='H', para='_200_euclidean_NA_dummy_add_0.5_intersect_16_GridEx')
plotResults('18-64_cpm',  dirName='H', para='_1000_euclidean_NA_dummy_add_0.5_intersect_80_GridEx9')
plotResults('2-5_cpm',    dirName='H', para='_200_euclidean_NA_dummy_add_0.5_intersect_120_GridEx14')
plotResults('2-8_cpm',    dirName='H', para='_2000_euclidean_NA_dummy_add_0.5_intersect_16_GridEx')
plotResults('T4857_cpm',  dirName='H', para='_50_euclidean_NA_dummy_add_0.5_intersect_80_GridEx9')


# Best Hybrid geom
print('Best Hybrid')  
plotResults('151507_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_80_GridEx9')
plotResults('151508_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_120_GridEx14')
plotResults('151509_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_200_GridEx24')
plotResults('151510_cpm', dirName='H', para='_1000_euclidean_NA_geom_lowf_add_0.5_intersect_120_GridEx14')
plotResults('151669_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_240_GridEx29')
plotResults('151670_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_240_GridEx29')
plotResults('151671_cpm', dirName='H', para='_200_euclidean_NA_geom_lowf_add_0.5_intersect_80_GridEx9')
plotResults('151672_cpm', dirName='H', para='_100_euclidean_NA_geom_lowf_add_0.5_intersect_160_GridEx19')
plotResults('151673_cpm', dirName='H', para='_1000_euclidean_NA_geom_lowf_add_0.5_intersect_16_GridEx')
plotResults('151674_cpm', dirName='H', para='_10_euclidean_NA_geom_lowf_add_0.5_intersect_8_Grid')
plotResults('151675_cpm', dirName='H', para='_10_euclidean_NA_geom_lowf_add_0.5_intersect_40_GridEx4')
plotResults('151676_cpm', dirName='H', para='_2000_euclidean_NA_geom_lowf_add_0.5_intersect_40_GridEx4')
plotResults('18-64_cpm',  dirName='H', para='_10_euclidean_NA_geom_lowf_add_0.5_intersect_240_GridEx29')
plotResults('2-5_cpm',    dirName='H', para='_100_euclidean_NA_geom_lowf_add_0.5_intersect_120_GridEx14')
plotResults('2-8_cpm',    dirName='H', para='_500_euclidean_NA_geom_lowf_add_0.5_intersect_80_GridEx9')
plotResults('T4857_cpm',  dirName='H', para='_10_euclidean_NA_geom_lowf_add_0.5_intersect_200_GridEx24')

# plotResults('151507_sctransform')
# plotResults('151508_sctransform')
# plotResults('151509_sctransform')
# plotResults('151510_sctransform')
# plotResults('151669_sctransform')
# plotResults('151670_sctransform')
# plotResults('151671_sctransform')
# plotResults('151672_sctransform')
# plotResults('151673_sctransform')
# plotResults('151674_sctransform')
# plotResults('151675_sctransform')
# plotResults('151676_sctransform')
# plotResults('18-64_sctransform')
# plotResults('2-5_sctransform')
# plotResults('2-8_sctransform')
# plotResults('T4857_sctransform')


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