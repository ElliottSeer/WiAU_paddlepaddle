#!/usr/bin/env python3.5
# -*- coding: UTF-8 -*-

import easygui as gui
import os
import numpy as np
import re
#import WiAU
user_data_path="/home/elliott/eudemon_server/process/user_data.dat"
all_user_path="/home/elliott/eudemon_server/user"

def usr_data_create(user_ID,name,passwd,reID):
    if isinstance(user_ID,list):
        user_ID=np.array(user_ID)
    if isinstance(name,list):
        name=np.array(name)
    if isinstance(passwd,list):
        passwd=np.array(passwd)
    if isinstance(reID,list):
        reID=np.array(reID)
    np.savez(all_user_path+'/'+str(user_ID)+"/"+str(user_ID)+"-info.npz",user_ID,name,passwd,reID)
    
def usr_data_read(ID):
    usr_data=np.load(all_user_path+"/"+str(ID)+'/'+str(ID)+'-info.npz')
    return usr_data["arr_0"],usr_data["arr_1"],usr_data["arr_2"],usr_data["arr_3"]
def mainwindow(user_ID):
    #_,user_name,user_passwd,user_reID=usr_data_read(user_ID)
    #Function=gui.indexbox(msg='ID:'+str(user_ID)+'\n'+'用户名:'+user_name[0],title='Eudemon',choices=("管理","训练","使用"))
    while(True):
        _,user_name,user_passwd,user_reID=usr_data_read(user_ID)
        msg=['ID:',str(user_ID),'\n','用户名:',str(user_name),'\n','关联ID:',str(user_reID)]
        msg=''.join(msg)
        Function=gui.indexbox(msg=msg,title='Eudemon',choices=("管理","训练","使用"))
        if Function==None:
            break
        if Function==0:
            if_del=gui.ccbox(msg='关联ID增删', title='Eudemon', choices=('删除', '增加'), image=None) 
            if if_del==1:
                do_del=0
                del_ID=gui.enterbox(msg="请输入删除ID",title='Eudemon')
                del_ID=int(del_ID)
                if del_ID==None:
                    break
                for i in user_reID.tolist():
                    if del_ID==i:
                        do_del+=1
                if do_del==0:
                    msg_del=gui.msgbox(msg="关联ID中无此ID",title='Eudemon')
                else:
                    user_reID=np.delete(user_reID,user_reID==del_ID)
                    user_reID=np.sort(user_reID)
                    usr_data_create(user_ID,user_name,user_passwd,user_reID)
                    msg_del=gui.msgbox(msg="已将此ID移除关联",title='Eudemon')
                    
            elif if_del==0:
                do_add=0
                add_ID=gui.enterbox(msg="请输入增加ID",title='Eudemon')
                add_ID=int(add_ID)

                ud_fp=open(user_data_path,'r')
                max_ID=int(ud_fp.readline())
                ud_fp.close()

                if add_ID==None:
                    break
                if add_ID>max_ID:
                    msg_add=gui.msgbox(msg="无此ID用户",title='Eudemon')
                    continue
                for i in user_reID.tolist():
                    if add_ID==i:
                        do_add+=1
                if do_add==0:
                    user_reID=np.append(user_reID,add_ID)
                    user_reID=np.sort(user_reID)
                    usr_data_create(user_ID,user_name,user_passwd,user_reID)
                    msg_add=gui.msgbox(msg="已将此ID添加关联",title='Eudemon')
                else:
                    msg_add=gui.msgbox(msg="此ID已经在关联ID中",title='Eudemon')
         #   Function=gui.indexbox(msg='ID:'+str(user_ID)+'\n'+'用户名:'+user_name[0],title='Eudemon',choices=("管理","训练","使用"))
        
        #elif Function==1:
            #Train

        #elif Function==2:
			#Infer


first=gui.buttonbox(title="Eudemon",image="Eudemon_logo.gif",choices=("说明","注册","登录"))
if first=='说明':
    gui.textbox(msg='注意事项',title='说明', text ='text', codebox = 1)
    first=gui.buttonbox(title="Eudemon",image="Eudemon_logo.gif",choices=("说明","注册","登录"))
    
elif first=='注册':
    msg=""
    title="新用户注册"
    fieldNames=["*用户名","*密码","*确认密码"]
    fieldValues=[]
    fieldValues=gui.multenterbox(msg,title,fieldNames)
    #print(fieldValues)
    while True:
        if fieldValues == None :
            break
        errmsg = ""
        for i in range(3):
            option = fieldNames[i].strip()
            if fieldValues[i].strip() == "" and option[0] == "*":
                errmsg += ("【%s】为必填项\n" %fieldNames[i])
        if fieldValues[1]!=fieldValues[2]:
            errmsg += ("密码不一致")
        if errmsg == "":
            break
        fieldValues = gui.multenterbox(errmsg,title,fieldNames,fieldValues)
    
    #print("您填写的资料如下:%s" %str(fieldValues))
    if fieldValues==None:
        os._exit(0)

    user_name=fieldValues[0]
    
    with open(user_data_path,'r+') as ud_fp:
        amount=int(ud_fp.readline())
        user_ID=amount+1
        user_data=user_name+'|'+str(user_ID)
        ud_fp.write(user_data+'\n')
        ud_fp.seek(0,0)
        ud_fp.write(str(user_ID))
        ud_fp.close()
    
    gui.msgbox('用户名：'+ user_name + '\n'+ ' ID'+ str(user_ID),'确认信息')
    
    os.makedirs(all_user_path+'/'+str(user_ID)+'/model')
    os.makedirs(all_user_path+'/'+str(user_ID)+'/train_dataset')
    os.makedirs(all_user_path+'/'+str(user_ID)+'/use_dataset')
    
    reID=[user_ID]
    usr_data_create(user_ID,user_name,fieldValues[1],reID)
    mainwindow(user_ID)
    
elif first=='登录':
    msg="请输入ID和密码"
    title="用户登录"
    fieldNames=["ID","密码"]
    if_success=0
    user_info=gui.multpasswordbox(msg,title,fieldNames)
    with open(user_data_path,'r+') as ud_fp:
        amount=ud_fp.readlines()
        Max_range=amount[0].strip()
        Max_range=int(Max_range)
        ud_fp.close()
    while True:
        errmsg = ""
        if user_info==None:
            break
        if user_info[0]=="":
            errmsg="请输入ID和密码"
            user_info = gui.multpasswordbox(errmsg,title,fieldNames,user_info)
            continue
        if user_info[0].isdigit():
            if int(user_info[0])<=0 or int(user_info[0])>Max_range:
                errmsg += ("此ID尚未创建")
            else:
                _,_,passwd,_=usr_data_read(int(user_info[0]))
                if passwd==user_info[1]:
                    #errmsg += ("登陆成功")
                    if_success=1
                    break
                else:
                    errmsg += ("密码错误")

        else:
            errmsg += ("ID必须仅由数字组成")

        #if errmsg == "":
        #    break
        user_info = gui.multpasswordbox(errmsg,title,fieldNames,user_info)
    if user_info==None:
        os._exit(0)
    elif if_success==1:
        if_success=0
        mainwindow(int(user_info[0]))

    







