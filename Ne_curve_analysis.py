#!/usr/bin/python
# -*- coding:UTF-8 -*-


'''
this project have two functions,
many psmc demographic trajectory results to predict their pattern or
one psmc demographic trajectory to find their pattern by compare with already exist data
'''


#this project is used to cluster through crest and trough
import sys,re
import getopt
import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn import mixture
from pandas import Series
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.externals import joblib


#deal files into broken lines
def read_files(arg):
	d_population_history = {}
	for v_species in arg:
		match_object = re.match(r'(\/.*\/)(.*)\.0\.txt', v_species)
		label = match_object.group(2)
		label = re.sub(r'scalled_', '', label)
		label = re.sub(r'_', ' ', label)
		a_sub_population = []
		a_broken_lines = []

		for line in open(v_species, 'r'):
			line = line.strip()
			line = line.split("\t")
			a_sub_population.append([float(line[0]), float(line[1])])
		for i in range(0, len(a_sub_population), 2):
			time = (a_sub_population[i][0] + a_sub_population[i+1][0])/2
			population_size = (a_sub_population[i][1] + a_sub_population[i+1][1])/2
			a_broken_lines.append([time, population_size])
		d_population_history.setdefault(label, a_broken_lines)
	return d_population_history


#get crest and trough in lines
def crest_trough(d_broken_lines):
	for key in d_broken_lines.keys():
		i = 1
		a_trend_v = []

		#increasing code as 1, decreasing code as -1
		while i < len(d_broken_lines[key]):
			if (d_broken_lines[key][i][1] - d_broken_lines[key][i-1][1]) > 0:
				a_trend_v.append(1)
			elif (d_broken_lines[key][i][1] - d_broken_lines[key][i-1][1]) < 0:
				a_trend_v.append(-1)
			else:
				a_trend_v.append(0)
			i +=1

		#deal the 0 part, there only exists increasing and decreasing trend,kick out ladder
		if a_trend_v[0] == 0 and a_trend_v[1] < 0:
			a_trend_v[0] == 1
		if a_trend_v[0] == 0 and a_trend_v[1] > 0:
			a_trend_v[0] == -1

		j = 1		
		while j < len(a_trend_v):
			if (a_trend_v[j]) >= 0 and (a_trend_v[j-1]) == 0:
				a_trend_v[j-1] = 1
			elif (a_trend_v[j]) < 0 and (a_trend_v[j-1]) == 0:
				a_trend_v[j-1] = -1
			j +=1
		
		#find crest and trough,-2 means crest,2 means trough
		k = 1
		a_fluctuation = []
		while k < len(a_trend_v):
			a_fluctuation.append(a_trend_v[k] - a_trend_v[k-1])
			k +=1
		a_diff = []
		
		#combine the time with crest and trough	
		for m in range(1, len(d_broken_lines[key]) -1):
			if a_fluctuation[m-1] == 2 or a_fluctuation[m-1] == -2:
				a_diff.append([d_broken_lines[key][m][0], a_fluctuation[m-1]])
			else:
				continue

		d_broken_lines[key] = a_diff
		
	return d_broken_lines


#Hopkins statistic
def hopkins_statistic(d_crest_trough):
	a_time = []
	for key in d_crest_trough.keys():
		for i in range(len(d_crest_trough[key])):
			if d_crest_trough[key][i][0] > 1047000 or d_crest_trough[key][i][0] < 10000:
				continue
			else:
				a_time.append(d_crest_trough[key][i][0])

	a_time = np.array(a_time).reshape(-1, 1)
	n = len(a_time)
	m = int(0.1 * n)
	nbrs = NearestNeighbors(n_neighbors=1, algorithm='brute').fit(a_time)

	rand_X = sample(range(0, n, 1), m)
	ujd = []
	wjd = []
	for j in range(0, m):
		u_dist, _ = nbrs.kneighbors(np.random.normal(size=(1, 1)).reshape(1, -1), 2, return_distance=True)
		ujd.append(u_dist[0][1])
		w_dist, _ = nbrs.kneighbors(a_time[rand_X[j]].reshape(1, -1), 2, return_distance=True)
		wjd.append(w_dist[0][1])
	
	H = sum(ujd) / (sum(ujd) + sum(wjd))

	return H
	

#Kmeans cluster
def Kmeans(d_crest_trough):
	a_time = []
	a_point = []
	a_species = []
	for key in d_crest_trough.keys():
		for i in range(len(d_crest_trough[key])):
			if d_crest_trough[key][i][0] > 1047000 or d_crest_trough[key][i][0] < 10000:
				continue
			else:
				a_time.append(d_crest_trough[key][i][0])
				a_point.append(d_crest_trough[key][i])
				a_species.append(key)
	a_time = np.array(a_time).reshape(-1, 1)
	kmeans = KMeans(n_clusters=7, random_state=0).fit(a_time)

	# evaluation by sihouette coefficient
	labels = kmeans.labels_
	cluster_centers = kmeans.cluster_centers_.tolist()
	silhouette_avg = silhouette_score(a_time, labels)
	sample_silhouette_values = silhouette_samples(a_time, labels).tolist()

	results = {}
	for i in range(7):
		for j in range(len(labels)):
			if i == labels[j]:
				results.setdefault(cluster_centers[i][0], []).append(sample_silhouette_values[j])
	
	for key in results.keys():
		results[key] = sum(results[key])/len(results[key])
	
	#reserve model
	joblib.dump(kmeans,'mammals.pkl')
	return silhouette_avg, results, a_point, labels, a_time, a_species, cluster_centers


#predict clusters
def predict(d_crest_trough):
	a_time = []
	for key in d_crest_trough.keys():
		for i in range(len(d_crest_trough[key])):
			if d_crest_trough[key][i][0] > 1100000:
				continue
			else:
				a_time.append([d_crest_trough[key][i][0]])

	kmeans = joblib.load('./mammals.pkl')
	cluster_centers = kmeans.cluster_centers_.tolist()
	predict_cluster = kmeans.predict(a_time)

	clusters = []	
	for i in predict_cluster:
		clusters.append(cluster_centers[i])

	return clusters

#running part
def final_model(analysis_type, arg):
	d_broken_lines = read_files(arg)
	d_crest_trough = crest_trough(d_broken_lines)

	if analysis_type == 'pattern':
		h_s = hopkins_statistic(d_crest_trough)
		silhouette_avg, cluster_silhouette_avg, a_point, labels, a_time, a_species, cluster_centers = Kmeans(d_crest_trough)
		

		#print("hopkins statistics is %s"%h_s)
		#print("average silhouette coefficient is %s"%silhouette_avg)
		#print("clusters average silhouette coefficient:")
		#for key in cluster_silhouette_avg.keys():
		#	print("%d: %s"%(key, cluster_silhouette_avg[key]))

		clusters = {}
		species_trough = {}
		species_crest = {}
		species = {}
		for i in range(0, 7):
			for j in range(len(labels)):
				if i == labels[j]:
					clusters.setdefault(i, []).append(a_point[j])
					species.setdefault(i, []).append(a_species[j])
					if a_point[j][1] == 2:
						species_trough.setdefault(i, []).append(a_species[j])
					if a_point[j][1] == -2:
						species_crest.setdefault(i, []).append(a_species[j])

		for key in clusters.keys():
			crest = 0
			trough = 0
			for i in clusters[key]:
				if i[1] == 2:
					trough +=1
				elif i[1] == -2:
					crest +=1
			#print("%d: trough,%d crest,%d"%(cluster_centers[key][0], trough, crest))
			print("%d"%cluster_centers[key][0])
			for i in range(len(clusters[key])):
				print(clusters[key][i][0], end = "\t")
			print("\n")
			#print(species[key]) 
			#print("crest:", "\n", species_crest[key])
			#print("trough:", "\n", species_trough[key])



	elif analysis_type == 'predict':
		clusters = predict(d_crest_trough)
		for i in range(len(clusters)):
			print("cluster center: %d"%clusters[i][0])
	else:
		print("python dtw_fluctuation.py -a pattern data or python dtw_fluctuation.py -a predict data")

if __name__ == '__main__':
	opt, arg = getopt.getopt(sys.argv[1:], "a:", ["analysis_type"])
	analysis_type = opt[0][1]

	final_model(analysis_type, arg)
