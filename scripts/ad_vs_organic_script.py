#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:31:42 2019

@author: archit
"""
import sys
import csv
import urllib.request as urllib2
from lxml import html, etree
import json
import pandas as pd
import math
import csv
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
import urllib.request
import traceback
import os
import urllib.request
import requests
import socket
import random
from multiprocessing.pool import Pool
from functools import partial
import isodate

def is_connected():
    try:
        socket.create_connection(("www.google.com", 80))
        return True
    except OSError:
        pass
    return False

keys = ["AIzaSyBLuKB_DU4NmGca6XGW5-BcYSxBB_2OmiA"]
key_index=0

class Youtube_extract:
    
    def __init__(self):
        self.hit= 0
        self.keys_list = ["AIzaSyCbZJKM6On5_gIX7wB31CntdxozKrDHiBA"]
    
    def get_channel_details(self,chan_ids_list,part='snippet',key="AIzaSyCbZJKM6On5_gIX7wB31CntdxozKrDHiBA"):
        url_c = "https://www.googleapis.com/youtube/v3/channels"
        responses = dict()
        for ind,chan in enumerate(chan_ids_list):
            try:
                querystring = {"id":chan ,"part":part,
                               "key":key}
                response = requests.request("GET", url_c, params=querystring)
                if response.json().get('error'):
                    responses.update({chnlid:[response,response.text]})
                    if response.json()['error']['errors']['reason']=='keyInvalid':
                        return [{chnlid:[response,response.text]}]
                    break
                responses[chan] = response.json()['items']
            except Exception as e:
                responses[chan] = {'error': [e,response,response.text]}
            if ind%100==0:
                print(ind)
        return (responses)

    def get_video_details(self,vid_ids_list,part='snippet',key="AIzaSyCbZJKM6On5_gIX7wB31CntdxozKrDHiBA"):
        url_v = "https://www.googleapis.com/youtube/v3/videos"
        responses = dict()
        for ind,vid in enumerate(vid_ids_list):
            try:
                querystring = {"id":vid ,"part":part,
                               "key":key}
                response = requests.request("GET", url_v, params=querystring)
                if response.json().get('error'):
                    responses.update({chnlid:[response,response.text]})
                    if response.json()['error']['errors']['reason']=='keyInvalid':
                        return [{chnlid:[response,response.text]}]
                    break
                responses[vid] = response.json()['items']
            except Exception as e:
                # responses[chan] = [e,response,response.text]
                responses[vid] = {'error': [e,response,response.text]}
            if ind%100==0:
                print(ind)
        return (responses)

    def playlist(self,channel_list,limit,part='contentDetails',key='AIzaSyCbZJKM6On5_gIX7wB31CntdxozKrDHiBA',only_id=1):
        playlist_url = 'https://www.googleapis.com/youtube/v3/playlistItems/'
        if limit<=50 and limit>0:
            maxResults=limit
        else:
            maxResults=50
        all_result = {}
        for chnlid in channel_list:
            vidcount = initial = 0
            nextPageToken =''
            results=[]
            # print('UU'+chnlid[2:])
            try:
                while nextPageToken or initial==0:
                    query = {
                        'playlistId':'UU'+chnlid[2:],
                        'part':part,
                        'key':key,
                        'pageToken':nextPageToken,
                        'maxResults':maxResults
                    }
                    response = requests.get(url = playlist_url,params = query)
                    # print(response,response.text)
                    if response.json().get('error'):
                        print(response.json())
#                         all_result.update({chnlid:[response,response.text]})
                        if response.json()['error'].get('errors'):
                            if response.json()['error']['errors'][0].get('reason'):
                                if response.json()['error']['errors'][0]['reason']=='keyInvalid':
                                    print("InvalidKey")
                                    return [{chnlid:{'error':[response,response.text]}}]
                        break
                    if limit==-1:
                        limit = response.json()['pageInfo']['totalResults']
                    # print(response,response.text)
                    
                    if only_id==1:
                        for i in range(response.json()['pageInfo']['resultsPerPage']):
                            try:
                                results.append(response.json()['items'][i]['contentDetails']['videoId'])
                            except:
                                pass
                    else:
                        results.append(response.json()['items'])
                    nextPageToken = response.json().get('nextPageToken')
                    vidcount = vidcount+ len(response.json()['items'])
                    if vidcount>=limit:
                        break
                    print("Completed:",vidcount)
                    
                    
                    initial = 1
                all_result.update({chnlid:results})
            except Exception as e:
                all_result[chnlid] = {'error': [e,traceback.print_exc(),response,response.text]}
                break
        return all_result

    def dnload_single_vidid_thumbnail(self, output_path, error_file_name, vidid):
        youtube_base = "https://i.ytimg.com/vi/"
        error_file = open(error_file_name, "a")
        not_found = 0
        # for vidid in vidid_list:
        try:
            urllib.request.urlretrieve(os.path.join(youtube_base, vidid+"/hqdefault.jpg"), os.path.join(output_path, vidid+".jpg"))
        except Exception as e:
            print("Error: ", e)
            error_file.write("{}\n".format(vidid))
            error_file.flush()
            not_found += 1
        # print("Total not found: ", not_found)
        error_file.close()
    
    def dnload_vidid_thumbnail(self, vidid_list, output_path, error_file_name):
        download = partial(self.dnload_single_vidid_thumbnail, output_path, error_file_name)
        with Pool(4) as p:
            p.map(download, vidid_list)


        # youtube_base = "https://i.ytimg.com/vi/"
        # error_file = open(error_file_name, "a")
        # not_found = 0
        # for vidid in vidid_list:
        #     try:
        #         urllib.request.urlretrieve(os.path.join(youtube_base, vidid+"/hqdefault.jpg"), os.path.join(output_path, vidid+".jpg"))
        #     except Exception as e:
        #         print("Error: ", e)
        #         error_file.write("{}\n".format(vidid))
        #         error_file.flush()
        #         not_found += 1
        # print("Total not found: ", not_found)
        # error_file.close()

    def all_channel_video_data(self,channel_list,limit,vid_part='snippet',output_path='../output/',error_file_name='../status/errors.txt'):
        # chnl_details_file = open(output_path+"channel_details.csv", "a")
        output_path += "thumbnails/"
        for i, chanlid in enumerate(channel_list):
            print("index: ", i, " : ", chanlid)
            # if i < 60:
            #     continue
            os.makedirs(output_path+chanlid, exist_ok=True)
            # all_result={}
            print("finding vidids")
            result = self.playlist([chanlid],limit)
            # print("###: ", result[chanlid])
            # exit(0)
            print("Downloading thumbnails")
            self.dnload_vidid_thumbnail(result[chanlid],output_path+chanlid,error_file_name)
            # print("finding channel meta")
            # all_result.update({chanlid:self.get_video_details(result[chanlid],part=vid_part)})
            # print("doing json dump")
            # json.dump(all_result, chnl_details_file)
            # chnl_details_file.write("\n")
        # return all_result


#if __name__ =='__main__':
#    data = Youtube_extract()
#    result = data.all_channel_video_data(['UCaJBo_nRSL1XF1eewwukTCg'],output_path='../data/',limit=-1,error_file_name='../error.txt')
#    # for chnlid in chnlid_list:
#    #     result = data.all_channel_video_data(['UCAGzeRhIE9NUASlH_X5MDEQ', 'UCAXN_awgRvcLdEIybqtOiEw'],limit=-1)
#        # print(len(result['UCAGzeRhIE9NUASlH_X5MDEQ']))
#        # json.dump(result,open(os.curdir+'/output.json','w+'))
#
#
## 

api_key = 'AIzaSyCrFWiPfGcb5IsyS-wpAMk6eaNdMaC8pXs'
channel = 'UCF0pVplsI8R5kcAqgtoRqoA'
vidStats =  'https://www.googleapis.com/youtube/v3/videos?part=id,statistics&id='
vidSnips = 'https://www.googleapis.com/youtube/v3/videos?part=id,snippet&id='
contentDetails = "https://www.googleapis.com/youtube/v3/videos?part=id,contentDetails&id="
channelStats = 'https://www.googleapis.com/youtube/v3/channels?part=snippet,statistics&id='








def get_previous_vid_details(vidId, chId):
    """
    """
    y_extract = Youtube_extract()
    channelList = [chId]
    playList = y_extract.playlist(channel_list=channelList, limit =-1)[chId]
    for i in range(0,len(playList)):
        if playList[i] == vidId:
            try:
                prevVidId = playList[i-1]
            except:
                 prevVidId = playList[i]
    firstTime = True
    fp = '/home/archit/Desktop/ad_vs_organic/'+str(prevVidId)+'validationVideoStats2.csv'
    print(fp)
    with open(fp, 'a',newline='') as c:
    
        writer = csv.writer(c)
        
        if firstTime:
            writer.writerow(['Id',  
                         'PrevLikeCount', 
                         'PrevDislikeCount',
                         'PrevViewCount', 
                         'PrevCommentCount', 
                         ])
    
        stats = json.load(urllib2.urlopen(vidStats + prevVidId + '&key=' + api_key))
        s = stats['items'][0]['statistics']
        try:
            writer.writerow([vidId, 
                         s['likeCount'],
                         s['dislikeCount'], 
                         s['viewCount'], 
                         s['commentCount']])
        except:
            print("Skipped video: " + str(vidId))
    c.close()
    dfPreVidSnippet = pd.read_csv(fp)
    return dfPreVidSnippet             
    
def get_channel_details(vidId, chId):
    """
    """
    channelInfo = list(chId.split())
    firstTime = True
    fp = '/home/archit/Desktop/ad_vs_organic/'+str(vidId)+ 'channelStats.csv'
    with open(fp, 'a',newline='') as c:
    
        writer = csv.writer(c)
        
        if firstTime:
            writer.writerow(['Channel Id', 
                         'publishedAt', 
                         'subscriberCount', 
                         'channelVideoCount', 
                         'channelViewCount', 
                         ])
        for channel in channelInfo:
            stats = json.load(urllib2.urlopen(channelStats + channel + '&key=' + api_key))
            
            writer.writerow([channel, 
                             stats['items'][0]['snippet']['publishedAt'],
                            stats['items'][0]['statistics']['subscriberCount'],
                             stats['items'][0]['statistics']['videoCount'],
                             stats['items'][0]['statistics']['viewCount']
                                                           ])
    c.close()
    dfChannelSnippet = pd.read_csv(fp)
    return dfChannelSnippet

def get_video_stats(vidId):
    """
    """
    firstTime = True
    fp ="/home/archit/Desktop/ad_vs_organic/"+str(vidId)+"validationVideoStats30.csv"
    with open(fp, 'a', newline='') as c:
    
        writer = csv.writer(c)
        
        if firstTime:
            writer.writerow(['Id', 
                         'Title', 
                         'Description', 
                         'LikeCount', 
                         'DislikeCount', 
                         'ViewCount', 
                         'FavoriteCount', 
                         'CommentCount', 
                         'PublishedAt', 
                         'Channel Id', 
                         'Channel Title',
                         'Duration', 'Definition'])
    
        vids = list(vidId.split())
        counter = 0;
        for vid in vids:
            stats = json.load(urllib2.urlopen(vidStats + vid + '&key=' + api_key))
            snips = json.load(urllib2.urlopen(vidSnips + vid + '&key=' + api_key))
            det = json.load(urllib2.urlopen(contentDetails + vid + '&key=' + api_key))
            if(len(stats['items'])==0):
                print("Could not find: " +  str(vid))
                continue
            s = stats['items'][0]['statistics']
            sn = snips['items'][0]
            details = det['items'][0]['contentDetails']
            print(sn['snippet']['channelId'])
#            try:
            writer.writerow([sn['id'], 
                      sn['snippet']['title'].encode('utf8'), 
                     sn['snippet']['description'].encode('utf8'), 
                     s['likeCount'],
                     s['dislikeCount'], 
                     s['viewCount'], 
                     s['favoriteCount'], 
                     s['commentCount'], 
                     sn['snippet']['publishedAt'],
                     sn['snippet']['channelId'],
                     sn['snippet']['channelTitle'],
                     details['duration'],
                     details['definition']])
#            except:
#                print("Skipped video: " + str(vid))
        c.close()
    dfVidSnippet = pd.read_csv(fp)
    return dfVidSnippet

def createFeatureMatrix(vidId):
    """function merges the dataFrame dependencies and return the final 
    input dataFrame and targets
    """
    dfVidSnippet = get_video_stats(vidId)
    chId = dfVidSnippet['Channel Id'][0]
    dur = isodate.parse_duration(dfVidSnippet['Duration'][0]).total_seconds()
    dfVidSnippet['Duration']=dur
    dfChannelDetails = get_channel_details(vidId, chId)
    dfPrevVidDetails = get_previous_vid_details(vidId, chId)

    

    # merge DataSet
    dfMergedData = dfVidSnippet.merge(dfPrevVidDetails, on = 'Id',  how = 'left')
#    print(dfMergedData['Channel Id_y'][0])
    dfMergedData = dfMergedData.merge(dfChannelDetails, on = 'Channel Id', how = 'left')
    return dfMergedData

def feature_engineer(dfData):
    dfData =dfData.drop(['FavoriteCount', 'Channel Title'],axis =1)
    
    dfData['Definition'] = dfData['Definition'].apply(lambda x: 1 if x=='hd' else 0)
    dfData['PublishedYear'] = dfData['PublishedAt'].apply(lambda x: x[:4])
    dfData['ChannelAge'] =  dfData['publishedAt'].apply(lambda x: x[:4])
    dfData['channelViewCount'] = np.log(dfData['channelViewCount'])
    dfData['LikeDislikeRatio'] = dfData['LikeCount']/(dfData['DislikeCount'] + dfData['LikeCount'])
    dfData['PrevCommentCount'] = dfData['PrevCommentCount'].fillna(0)
    dfData['PrevDislikeCount'] = dfData['PrevDislikeCount'].fillna(0)
    dfData['PrevLikeCount'] = dfData['PrevLikeCount'].fillna(0)
    dfData['PrevViewCount'] = dfData['PrevViewCount'].fillna(0)   
    dfData['duration'] = dfData['Duration'].fillna(0)
    dfData['LikeDislikeRatio'] = dfData['LikeDislikeRatio'].replace(np.inf, np.nan)
    dfData['LikeDislikeRatio'] = dfData['LikeDislikeRatio'].fillna(0)
    dfData['Channel subscriberCount'] = dfData['subscriberCount'].fillna(0)
    dfData = dfData.drop(['Title','Definition','Duration','Description','PublishedAt','channelViewCount','publishedAt','Channel Id'],axis = 1)
    
    return dfData


def main(vidId):
    """trains current video id using LSTM model to predict ad views and organic views
    Args:
        vidId - video id of youtube video
    Returns:
        a tuple of (adViews, organicViews)
    """
    dfValidate = createFeatureMatrix(vidId)
    df = feature_engineer(dfValidate)
    X, y = df.drop('ViewCount',axis = 1) , np.log(df['ViewCount'])
    load_model = pickle.load(open("/home/archit/Desktop/ad_vs_organic/Gbr005.pickle.dat",'rb'))
    X_id, X = X.Id, X.drop('Id', axis = 1)
    dfValidate['Pred'] = np.exp(np.array(load_model.predict(X)))
    dfValidate['Diff'] = dfValidate['ViewCount'] - dfValidate['Pred']
    dfValidate['Predicted Ad% Lower Bound'] = (dfValidate['Diff']-800)/dfValidate['ViewCount']*100
    dfValidate['Predicted Ad% upper Bound'] = (dfValidate['Diff']+800)/dfValidate['ViewCount']*100
    dfValidate['Predicted Ad %'] = (dfValidate['Diff'])/dfValidate['ViewCount']*100
    return (dfValidate['Predicted Ad% Lower Bound'][0], dfValidate['Predicted Ad %'][0], dfValidate['Predicted Ad% upper Bound'][0])
    
#print(main('u2weEofBqGg'))
if __name__=='__main__':   
    chId = sys.argv[1]  
    print(main(chId))                                                                                                                               