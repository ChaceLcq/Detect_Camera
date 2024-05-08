package com.example.hk.onecamera;


import java.util.ArrayList;

import MvCameraControlWrapper.CameraControlException;
import MvCameraControlWrapper.CameraEventCallBack;
import MvCameraControlWrapper.CameraExceptionCallBack;
import MvCameraControlWrapper.CameraImageCallBack;
import MvCameraControlWrapper.MvCameraControl;
import MvCameraControlWrapper.MvCameraControlDefines;

/**
 * Created by panfeilong on 2019/12/3.
 */

public class CameraManager {
    private ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> deviceList = new ArrayList<>();
    private MvCameraControlDefines.Handle handle = null;
    private final static int MV_OK = 0;


    public String GetSDKVersion() {
        return MvCameraControl.MV_CC_GetSDKVersion();
    }

    public int EnumerateTls() {
        return MvCameraControl.MV_CC_EnumerateTls();
    }

    public boolean IsDeviceAccessible(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo, int nAccessMode) {
        return MvCameraControl.MV_CC_IsDeviceAccessible(stDeviceInfo, nAccessMode);
    }

    public ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> enumDevice() throws CameraControlException {
        deviceList = MvCameraControl.MV_CC_EnumDevices(MvCameraControlDefines.MV_USB_DEVICE | MvCameraControlDefines.MV_GIGE_DEVICE);
        return deviceList;
    }

    public void createHandle(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo) throws CameraControlException {
        if (handle != null) {
            int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
            handle = null;
        }
        handle = MvCameraControl.MV_CC_CreateHandle(stDeviceInfo);
    }

    public void createHandle(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo, boolean flag) throws CameraControlException {
        if (handle != null) {
            int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        }
        handle = MvCameraControl.MV_CC_CreateHandle(stDeviceInfo, flag);
    }

    public void destroyHandle() {
        if (handle != null) {
            int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
            handle = null;
        }
    }

    public int openDevice() {
        return MvCameraControl.MV_CC_OpenDevice(handle);
    }


    public int startDevice() {
        return MvCameraControl.MV_CC_StartGrabbing(handle);
    }

    public int getOneFrameTimeout(byte[] datas, MvCameraControlDefines.MV_FRAME_OUT_INFO info, int time) {
        return MvCameraControl.MV_CC_GetBitmapTimeout(handle, datas, info, time);
    }

    public void setImageNodeNumBt(int num) {
        int nRet = MvCameraControl.MV_CC_SetImageNodeNum(handle, num);
    }

    public int stopDevice() {
        return MvCameraControl.MV_CC_StopGrabbing(handle);
    }

    public int closeDevice() {
        return MvCameraControl.MV_CC_CloseDevice(handle);
    }

    public int registerExceptionCallBackBt(CameraExceptionCallBack callBack) {
        return MvCameraControl.MV_CC_RegisterExceptionCallBack(handle, callBack);
    }

    public int registerImageCallBack(CameraImageCallBack callBack) {
        return MvCameraControl.MV_CC_RegisterImageCallBack(handle, callBack);
    }

    public int registerEventCallBack(String eventName, CameraEventCallBack callBack) {
        return MvCameraControl.MV_CC_RegisterEventCallBack(handle, eventName, callBack);
    }

    public int getOptimalPacketSize() {
        return MvCameraControl.MV_CC_GetOptimalPacketSize(handle);
    }

    public int getNodeAccessMode(String key, MvCameraControlDefines.MV_XML_AccessMode accessMode) {
        MvCameraControlDefines.MV_XML_AccessMode result = null;
        try {
            result = MvCameraControl.MV_XML_GetNodeAccessMode(handle, key);
        } catch (Exception e) {
            
        }
        return result.getnValue();
    }


    public int fileAccessWrite(MvCameraControlDefines.MV_CC_FILE_ACCESS access) {
        return MvCameraControl.MV_CC_FileAccessWrite(handle, access);
    }

    public int GIGE_setResend(int bEnable) {
        return MvCameraControl.MV_GIGE_SetResend(handle, bEnable);
    }

    public int GIGE_setResend(int bEnable, int nMaxResendPercent, int nResendTimeout) {
        return MvCameraControl.MV_GIGE_SetResend(handle, bEnable, nMaxResendPercent, nResendTimeout);
    }

    public int featureSave(String name) {
        return MvCameraControl.MV_CC_FeatureSave(handle, name);
    }

    public int featureLoad(String name) {
        return MvCameraControl.MV_CC_FeatureLoad(handle, name);
    }

    public int getNetTransInfoBt(MvCameraControlDefines.MV_NETTRANS_INFO info) {
        return MvCameraControl.MV_GIGE_GetNetTransInfo(handle, info);
    }

    public int getAllMatchInfo(MvCameraControlDefines.MV_ALL_MATCH_INFO info) {
        return MvCameraControl.MV_CC_GetAllMatchInfo(handle, info);
    }

    public int getGenICamXML(byte[] bytes, Integer integer) {
        return MvCameraControl.MV_XML_GetGenICamXML(handle, bytes, integer);
    }

    public int getIntValue(String key, Integer value) {
        return MvCameraControl.MV_CC_GetIntValue(handle, key, value);
    }

    public int getIntValue(String key, MvCameraControlDefines.MVCC_INTVALUE intvalue) {
        return MvCameraControl.MV_CC_GetIntValue(handle, key, intvalue);
    }

    public int setIntValue(String key, long data) {
        return MvCameraControl.MV_CC_SetIntValue(handle, key, data);
    }

    public int getFloatValue(String key, MvCameraControlDefines.MVCC_FLOATVALUE floatvalue) {
        return MvCameraControl.MV_CC_GetFloatValue(handle, key, floatvalue);
    }

    public int getFloatValue(String key, Float floatvalue) {
        return MvCameraControl.MV_CC_GetFloatValue(handle, key, floatvalue);
    }

    public int setFloatValue(String key, float floatvalue) {
        return MvCameraControl.MV_CC_SetFloatValue(handle, key, floatvalue);
    }

    public int getEnumValue(String key, Integer value) {
        return MvCameraControl.MV_CC_GetEnumValue(handle, key, value);
    }

    public int getEnumValue(String key, MvCameraControlDefines.MVCC_ENUMVALUE intvalue) {
        return MvCameraControl.MV_CC_GetEnumValue(handle, key, intvalue);
    }

    public int setEnumValue(String key, int value) {
        return MvCameraControl.MV_CC_SetEnumValue(handle, key, value);
    }

    public int setEnumValueByString(String key, String value) {
        return MvCameraControl.MV_CC_SetEnumValueByString(handle, key, value);
    }


    public int getBoolValue(String key, Boolean bValue) {
        int nRet = MvCameraControl.MV_CC_GetBoolValue(handle, key, bValue);
        return nRet;
    }

    public int setBoolValue(String key, boolean bValue) {
        int nRet = MvCameraControl.MV_CC_SetBoolValue(handle, key, bValue);
        return nRet;
    }

    public int getStrValue(String key, MvCameraControlDefines.MVCC_STRINGVALUE stringvalue) {
        return MvCameraControl.MV_CC_GetStringValue(handle, key, stringvalue);
    }

    public int setStrValu(String key, String value) {
        return MvCameraControl.MV_CC_SetStringValue(handle, key, value);
    }

    public int setCommandValue(String key) {
        return MvCameraControl.MV_CC_SetCommandValue(handle, key);
    }

    public int setGvspTimeoutBt(int time) {
        int nRet = MvCameraControl.MV_GIGE_SetGvspTimeout(handle, time);
        return nRet;
    }

    public int getGvspTimeoutBt(Integer value) {
        int nRet = MvCameraControl.MV_GIGE_GetGvspTimeout(handle, value);
        return nRet;
    }

    public int setResendMaxRetryTimes(int value) {
        int nRet = MvCameraControl.MV_GIGE_SetResendMaxRetryTimes(handle, value);
        return nRet;
    }

    public int getResendMaxRetryTimes(Integer value) {
        int nRet = MvCameraControl.MV_GIGE_GetResendMaxRetryTimes(handle, value);
        return nRet;
    }

    public int setResendTimeInterval(int value) {
        int nRet = MvCameraControl.MV_GIGE_SetResendTimeInterval(handle, value);
        return nRet;
    }

    public int getResendTimeInterval(Integer value) {
        int nRet = MvCameraControl.MV_GIGE_GetResendTimeInterval(handle, value);
        return nRet;
    }

    public int convertPixelType(MvCameraControlDefines.MV_CC_PIXEL_CONVERT_PARAM param) {
        return MvCameraControl.MV_CC_ConvertPixelType(handle, param);
    }


}
