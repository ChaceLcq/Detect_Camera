package com.example.hk.basemodule;

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
    public String cameraPosition = "";
    private ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> deviceList = new ArrayList<>();
    private MvCameraControlDefines.Handle handle;
    //    private IMainView iMainView;
    private final static int MV_OK = 0;

    public CameraManager(IMainView iMainView) {
//        this.iMainView = iMainView;
    }

    public String GetSDKVersion() {
        return MvCameraControl.MV_CC_GetSDKVersion();
    }

    public int EnumerateTls() {
        return MvCameraControl.MV_CC_EnumerateTls();
    }

    public boolean IsDeviceAccessible(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo, int nAccessMode) {
        return MvCameraControl.MV_CC_IsDeviceAccessible(stDeviceInfo, nAccessMode);
    }

    public ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> enumDevice() {
        try {
            deviceList = MvCameraControl.MV_CC_EnumDevices(MvCameraControlDefines.MV_USB_DEVICE | MvCameraControlDefines.MV_GIGE_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
//            iMainView.showLog(e.errMsg + ":" + e.errCode);
        }
        return deviceList;
    }

    public int createHandle(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo) {
        if (handle != null) {
            int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        }
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDeviceInfo);
            if (handle == null) {
                return -1;
            } else {
                return MV_OK;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return e.errCode;
//            iMainView.showLog(e.errMsg + e.errCode);
        }
    }

    public int destroyHandle() {
        int nRet = -1;
        if (handle != null) {
            nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
            handle = null;
        }
        return nRet;
    }

    public void createHandle(MvCameraControlDefines.MV_CC_DEVICE_INFO stDeviceInfo, boolean flag) {
        if (handle != null) {
            int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
//            iMainView.showLog("DestroyHandle：" + nRet);
        }
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDeviceInfo, flag);
//            iMainView.showLog("CreateHandle 无日志");
        } catch (CameraControlException e) {
            e.printStackTrace();
//            iMainView.showLog(e.errMsg + e.errCode);
        }
    }

    public int openDevice() {
        if (handle == null) {
//            iMainView.showLog("请先创建句柄");
            return -1;
        }
        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (MV_OK != nRet) {
//            iMainView.showLog("打开设备失败:" + Integer.toHexString(nRet));
        } else {
//            iMainView.showLog("打开设备成功");
        }
        return nRet;
    }


    public int startGrabbing() {
        int nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        return nRet;
    }

    public int getOneFrameTimeout(byte[] datas, MvCameraControlDefines.MV_FRAME_OUT_INFO info, int time) {
        int nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(handle, datas, info, time);
        return nRet;
    }
    public int getBitmapTimeout(byte[] datas, MvCameraControlDefines.MV_FRAME_OUT_INFO info, int time) {
        int nRet = MvCameraControl.MV_CC_GetBitmapTimeout(handle, datas, info, time);
        return nRet;
    }

    public void setImageNodeNumBt(int num) {
        int nRet = MvCameraControl.MV_CC_SetImageNodeNum(handle, num);
//        iMainView.showLog("SetImageNodeNum:" + nRet);
    }

    public int stopGrabbing() {

        int nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        return nRet;
    }

    public int closeDevice() {
        int nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        return nRet;
    }

    public int registerExceptionCallBack(CameraExceptionCallBack callBack) {
        int nRet = MvCameraControl.MV_CC_RegisterExceptionCallBack(handle, callBack);
//        iMainView.showLog("注册回调函数:" + nRet);
        return nRet;
    }

    public int registerImageCallBack(CameraImageCallBack callBack) {
        int nRet = MvCameraControl.MV_CC_RegisterImageCallBack(handle, callBack);
        return nRet;
    }

    public int registerEventCallBack(String eventName, CameraEventCallBack callBack) {
        int nRet = MvCameraControl.MV_CC_RegisterEventCallBack(handle, eventName, callBack);
        return nRet;
    }

    public int getOptimalPacketSize() {
        return MvCameraControl.MV_CC_GetOptimalPacketSize(handle);
    }

    public int fileAccessWrite(MvCameraControlDefines.MV_CC_FILE_ACCESS access) {
        int nRet = MvCameraControl.MV_CC_FileAccessWrite(handle, access);
//        iMainView.showLog("FileAccessWrite:" + Integer.toHexString(nRet));
        return nRet;
    }

    public int GIGE_setResend(int bEnable) {
        int nRet = MvCameraControl.MV_GIGE_SetResend(handle, bEnable);
//        iMainView.showLog("SetResend" + nRet);
        return nRet;
    }

    public int GIGE_setResend(int bEnable, int nMaxResendPercent, int nResendTimeout) {
        int nRet = MvCameraControl.MV_GIGE_SetResend(handle, bEnable, nMaxResendPercent, nResendTimeout);
//        iMainView.showLog("SetResend" + nRet);
        return nRet;
    }

    public int featureSave(String name) {
        int nRet = MvCameraControl.MV_CC_FeatureSave(handle, name);
//        iMainView.showLog("FeatureSave" + Integer.toHexString(nRet));
        return nRet;
    }

    public int featureLoad(String name) {
        int nRet = MvCameraControl.MV_CC_FeatureLoad(handle, name);
//        iMainView.showLog("FeatureLoad" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getNetTransInfo(MvCameraControlDefines.MV_NETTRANS_INFO info) {
        int nRet = MvCameraControl.MV_GIGE_GetNetTransInfo(handle, info);
//        iMainView.showLog("GetNetTransInfo:" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getAllMatchInfo(MvCameraControlDefines.MV_ALL_MATCH_INFO info) {
        int nRet = MvCameraControl.MV_CC_GetAllMatchInfo(handle, info);
//        iMainView.showLog("GetAllMatchInfo:" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getGenICamXML(byte[] bytes, Integer integer) {
        int nRet = MvCameraControl.MV_XML_GetGenICamXML(handle, bytes, integer);
//        iMainView.showLog("GetGenICamXML:" + Integer.toHexString(nRet) + "buffer size :" + integer);
        return nRet;
    }

    public int forceIp(String ip, String subNetMask, String defaultGateWay) {
        return MvCameraControl.MV_GIGE_ForceIp(handle, ip, subNetMask, defaultGateWay);
    }

    public int getIntValue(String key, Integer value) {
        int nRet = MvCameraControl.MV_CC_GetIntValue(handle, key, value);
        return nRet;
    }

    public int getIntValue(String key, MvCameraControlDefines.MVCC_INTVALUE intvalue) {
        int nRet = MvCameraControl.MV_CC_GetIntValue(handle, key, intvalue);
//        iMainView.showLog("GetIntValue" +nRet);
        return nRet;
    }

    public int setIntValue(String key, long data) {
        int nRet = MvCameraControl.MV_CC_SetIntValue(handle, key, data);
//        iMainView.showLog("GetIntValue" +nRet);
        return nRet;
    }


    public int getEnumValue(String key, Integer value) {
        int nRet = MvCameraControl.MV_CC_GetEnumValue(handle, key, value);

//        iMainView.showLog("GetEnumValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getEnumValue(String key, MvCameraControlDefines.MVCC_ENUMVALUE intvalue) {
        int nRet = MvCameraControl.MV_CC_GetEnumValue(handle, key, intvalue);
//        iMainView.showLog("GetEnumValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int setEnumValue(String key, int value) {
        int nRet = MvCameraControl.MV_CC_SetEnumValue(handle, key, value);
        return nRet;
    }
    public int setEnumValueByString(String key, String value) {
        int nRet = MvCameraControl.MV_CC_SetEnumValueByString(handle, key, value);
        return nRet;
    }

    public int getValueFloat(String key, Float fValue) {
        int nRet = MvCameraControl.MV_CC_GetFloatValue(handle, key, fValue);
//        iMainView.showLog("FloatValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getValueFloat(String key, MvCameraControlDefines.MVCC_FLOATVALUE floatvalue) {
        int nRet = MvCameraControl.MV_CC_GetFloatValue(handle, key, floatvalue);
//        iMainView.showLog("FloatValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int setValueFloat(String key, float floatvalue) {
        int nRet = MvCameraControl.MV_CC_SetFloatValue(handle, key, floatvalue);
//        iMainView.showLog("FloatValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getValueBool(String key, Boolean bValue) {
        int nRet = MvCameraControl.MV_CC_GetBoolValue(handle, key, bValue);
//        iMainView.showLog("FloatValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int setValueBool(String key, Boolean bValue) {
        int nRet = MvCameraControl.MV_CC_SetBoolValue(handle, key, bValue);
//        iMainView.showLog("FloatValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int getValueStr(String key, MvCameraControlDefines.MVCC_STRINGVALUE stringvalue) {
        int nRet = MvCameraControl.MV_CC_GetStringValue(handle, key, stringvalue);
//        iMainView.showLog("GetStringValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int setValueStr(String key, String value) {
        int nRet = MvCameraControl.MV_CC_SetStringValue(handle, key, value);
//        iMainView.showLog("GetStringValue" + Integer.toHexString(nRet));
        return nRet;
    }

    public int setCommandValue(String key) {
        int nRet = MvCameraControl.MV_CC_SetCommandValue(handle, key);
        return nRet;
    }

}
