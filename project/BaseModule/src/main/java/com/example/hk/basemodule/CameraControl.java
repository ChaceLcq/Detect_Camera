package com.example.hk.basemodule;

import android.util.Log;

import java.util.ArrayList;

import MvCameraControlWrapper.CameraControlException;
import MvCameraControlWrapper.CameraEventCallBack;
import MvCameraControlWrapper.CameraExceptionCallBack;
import MvCameraControlWrapper.CameraImageCallBack;
import MvCameraControlWrapper.MvCameraControl;
import MvCameraControlWrapper.MvCameraControlDefines;

import static MvCameraControlWrapper.MvCameraControlDefines.MV_GIGE_DEVICE;
import static MvCameraControlWrapper.MvCameraControlDefines.MV_USB_DEVICE;
import static MvCameraControlWrapper.MvCameraControlDefines.MvGvspPixelType.PixelType_Gvsp_Mono8;

/**
 * Created by panfeilong on 2019/12/13.
 */

public class CameraControl {
    public void GetSDKVersion() {
        String version = MvCameraControl.MV_CC_GetSDKVersion();
    }
    public void EnumerateTls() {
        int nTransLayers = MvCameraControl.MV_CC_EnumerateTls();
        if (nTransLayers == MvCameraControlDefines.MV_GIGE_DEVICE) {
            Log.e("CameraControl", "GigeDevice");
        } else if (nTransLayers == MvCameraControlDefines.MV_USB_DEVICE) {
            Log.e("CameraControl", "UsbDevice");
        } else if (nTransLayers == (MvCameraControlDefines.MV_GIGE_DEVICE + MvCameraControlDefines.MV_USB_DEVICE)) {
            Log.e("CameraControl", "GigeDevice and UsbDevice");
        }
    }
    public void EnumDevices() {
        try {
            ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = MvCameraControl.
                    MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
            Log.e("CameraControl", e.errCode + e.errMsg);
        }
    }
    public void IsDeviceAccessible() {
        try {
            ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
            if (devicesList == null) {
                Log.e("CameraControl", "error: EnumDevices fail");
                return;
            }
            //默认使用第一个设备，实际开发选择自己的目标设备
            boolean falg = MvCameraControl.MV_CC_IsDeviceAccessible(devicesList.get(0), MvCameraControlDefines.MV_ACCESS_Exclusive);
            Log.e("CameraControl", "isDeviceAccessible:" + falg);

        } catch (CameraControlException e) {
            e.printStackTrace();
            Log.e("CameraControl", "error: EnumDevices fail nRet" + e.errCode);
        }
    }
    public void CreateHandle() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        MvCameraControlDefines.Handle handle = null;
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            Log.e("CameraControl", e.errCode + e.errMsg);
            return;
        }
        int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);

    }
    public void DestroyHandle() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            Log.e("CameraControl", "error: CreateHandle fail:" + e.errCode);
            return;
        }

        int nRet = MvCameraControl.MV_CC_DestroyHandle(handle);

    }
    public void OpenDevice() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
    }
    public void CloseDevice() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
    }
    public void RegisterImageCallBack() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_RegisterImageCallBack(handle, new CameraImageCallBack() {
            @Override
            public int OnImageCallBack(byte[] bytes, MvCameraControlDefines.MV_FRAME_OUT_INFO mv_frame_out_info) {
                return 0;
            }
        });
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: RegisterImageCallBack fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        try {
            Thread.sleep(100000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
    }
    public void StartGrabbing() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }


        try {
            Thread.sleep(100000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
    }
    public void StopGrabbing() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        try {
            Thread.sleep(100000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
    }
    public void GetOneFrameTimeout() {

        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
        nRet = MvCameraControl.MV_CC_GetIntValue(handle, "PayloadSize", intvalue);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: PayloadSize fail");
            return;
        }

        byte[] datas = new byte[(int) intvalue.curValue + 2048];
        MvCameraControlWrapper.MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();
        for (int i = 0; i < 50; i++) {
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(handle, datas, info, 1000);
            if (nRet != MvCameraControlDefines.MV_OK) {
                Log.e("CameraControl", "error: GetOneFrameTimeout fail");
                return;
            }
        }


        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);


    }
    public void SetImageNodeNum() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_SetImageNodeNum(handle, 1);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetImageNodeNum fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
        nRet = MvCameraControl.MV_CC_GetIntValue(handle, "PayloadSize", intvalue);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: PayloadSize fail");
            return;
        }

        byte[] datas = new byte[(int) intvalue.curValue + 2048];
        MvCameraControlWrapper.MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();
        for (int i = 0; i < 50; i++) {
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(handle, datas, info, 1000);
            if (nRet != MvCameraControlDefines.MV_OK) {
                Log.e("CameraControl", "error: GetOneFrameTimeout fail");
                return;
            }
        }


        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);


    }
    public void RegisterExceptionCallBack() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_RegisterExceptionCallBack(handle, new CameraExceptionCallBack() {
            @Override
            public int OnExceptionCallBack(int i) {
                return 0;
            }
        });
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: RegisterExceptionCallBack fail");
            return;
        }


        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
        nRet = MvCameraControl.MV_CC_GetIntValue(handle, "PayloadSize", intvalue);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: PayloadSize fail");
            return;
        }

        byte[] datas = new byte[(int) intvalue.curValue + 2048];
        MvCameraControlWrapper.MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();
        for (int i = 0; i < 50; i++) {
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(handle, datas, info, 1000);
            if (nRet != MvCameraControlDefines.MV_OK) {
                Log.e("CameraControl", "error: GetOneFrameTimeout fail");
                return;
            }
        }


        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);

    }
    public void GetOptimalPacketSize() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        int nPacketSize = MvCameraControl.MV_CC_GetOptimalPacketSize(handle);
        if (nPacketSize > 0) {
            nRet = MvCameraControl.MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
            if (nRet != MvCameraControlDefines.MV_OK) {
                Log.e("CameraControl", "Warning: Set Packet Size fail nRet" + nRet);
                return;
            }
        } else {
            Log.e("CameraControl", "Warning: Get Packet Size fail nRet" + nPacketSize);
            return;
        }


        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    public void FileAccessRead() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        MvCameraControlDefines.MV_CC_FILE_ACCESS file_access = new MvCameraControlDefines.MV_CC_FILE_ACCESS();
        file_access.userFileName = "UserSet1.txt";
        file_access.devFileName = "UserSet1";
        nRet = MvCameraControl.MV_CC_FileAccessRead(handle, file_access);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: FileAccessRead fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    public void FileAccessWrite() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        MvCameraControlDefines.MV_CC_FILE_ACCESS file_access = new MvCameraControlDefines.MV_CC_FILE_ACCESS();
        file_access.userFileName = "UserSet1.txt";
        file_access.devFileName = "UserSet1";
        nRet = MvCameraControl.MV_CC_FileAccessWrite(handle, file_access);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: FileAccessRead fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    public void RegisterEventCallBack() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_RegisterEventCallBack(handle, "ExposureEnd", new CameraEventCallBack() {
            @Override
            public int OnEventCallBack(MvCameraControlDefines.MV_EVENT_OUT_INFO mv_event_out_info) {
                return 0;
            }
        });

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: RegisterEventCallBack fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }

    public void SaveImage() {       
    }
    public void FeatureSaveAndLoad() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }

        nRet = MvCameraControl.MV_CC_FeatureSave(handle, "FeatureFile.ini");
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "Save Feature fail! nRet" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_FeatureLoad(handle, "FeatureFile.ini");
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "Load Feature fail! nRet" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "ClosDevice fail! nRet " + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "Destroy Handle fail! nRet " + nRet);
            return;
        }
    }
    public void GetNetTransInfo() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
        nRet = MvCameraControl.MV_CC_GetIntValue(handle, "PayloadSize", intvalue);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: PayloadSize fail");
            return;
        }

        byte[] datas = new byte[(int) intvalue.curValue + 2048];
        MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();
        for (int i = 0; i < 50; i++) {
            nRet = MvCameraControl.MV_CC_GetOneFrameTimeout(handle, datas, info, 1000);
            if (nRet != MvCameraControlDefines.MV_OK) {
                Log.e("CameraControl", "error: GetOneFrameTimeout fail");
                return;
            }
        }
        MvCameraControlDefines.MV_NETTRANS_INFO info1 = new MvCameraControlDefines.MV_NETTRANS_INFO();
        nRet = MvCameraControl.MV_GIGE_GetNetTransInfo(handle, info1);
        if (nRet == MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "GetDataSize" + info1.reviceDataSize);
        }


        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StopGrabbing fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);

    }
    public void GetAllMatchInfo() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: StartGrabbing fail");
            return;
        }

        try {
            Thread.sleep(5000);
        } catch (InterruptedException e) {
            e.printStackTrace();
        }
        MvCameraControlDefines.MV_ALL_MATCH_INFO info = new MvCameraControlDefines.MV_ALL_MATCH_INFO();
        nRet = MvCameraControl.MV_CC_GetAllMatchInfo(handle, info);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetAllMatchInfo fail");
            return;
        }
        nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "Operation: Stop Grabbing......failed!");
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail " + nRet);
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail " + nRet);
            return;
        }

    }
    public void GetGenICamXML() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail");
            return;
        }
        int MAX_XML_FILE_SIZE = (1024 * 1024 * 3);

        byte[] data = new byte[MAX_XML_FILE_SIZE];
        Integer nXMLDataLen = new Integer(0);
        nRet = MvCameraControl.MV_XML_GetGenICamXML(handle, data, nXMLDataLen);
        if (MvCameraControlDefines.MV_OK != nRet || nXMLDataLen > MAX_XML_FILE_SIZE) {
            Log.e("CameraControl", "error: GetGenICamXML failed! " + nRet);
            return;
        }
        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail " + nRet);
            return;
        }
        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail " + nRet);
            return;
        }
    }
    public void ForceIp() {

    }
    public void SetIpConfig() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }


        int nRet = MvCameraControl.MV_GIGE_SetIpConfig(handle, MvCameraControlDefines.MVCC_IP_CONFIG.MV_IP_CFG_LLA);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetIpConfig fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void GetIntValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }

        MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
        nRet = MvCameraControl.MV_CC_GetIntValue(handle, "Width", intvalue);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetIntValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetIntValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetIntValue(handle, "Width", 1000);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetIntValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void GetEnumValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }

        MvCameraControlDefines.MVCC_ENUMVALUE enumvalue = new MvCameraControlDefines.MVCC_ENUMVALUE();
        nRet = MvCameraControl.MV_CC_GetEnumValue(handle, "GainAuto", enumvalue);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetEnumValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetEnumValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetEnumValue(handle, "GainAuto", 0);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetEnumValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetEnumValueByString() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetEnumValueByString(handle, "GainAuto", "Off");

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetEnumValueByString fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void GetFloatValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }

        MvCameraControlDefines.MVCC_FLOATVALUE floatvalue = new MvCameraControlDefines.MVCC_FLOATVALUE();
        nRet = MvCameraControl.MV_CC_GetFloatValue(handle, "Gain", floatvalue);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetFloatValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetFloatValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetFloatValue(handle, "Gain", 0f);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetFloatValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void GetBoolValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }

        Boolean booleanvalue = new Boolean(false);
        nRet = MvCameraControl.MV_CC_GetBoolValue(handle, "Gain", booleanvalue);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetBoolValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetBoolValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetBoolValue(handle, "Gain", false);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetBoolValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void GetStringValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }

        MvCameraControlDefines.MVCC_STRINGVALUE stringvalue = new MvCameraControlDefines.MVCC_STRINGVALUE();
        nRet = MvCameraControl.MV_CC_GetStringValue(handle, "DeviceUserID", stringvalue);

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: GetStringValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetStringValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetStringValue(handle, "DeviceUserID", "hikCamera");

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetStringValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
    void SetCommandValue() {
        ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> devicesList = null;
        try {
            devicesList = MvCameraControl.MV_CC_EnumDevices(MV_GIGE_DEVICE | MV_USB_DEVICE);
        } catch (CameraControlException e) {
            e.printStackTrace();
        }
        if (devicesList == null) {
            Log.e("CameraControl", "error: EnumDevices fail");
            return;
        }
        MvCameraControlDefines.Handle handle = null;
        MvCameraControlDefines.MV_CC_DEVICE_INFO stDevInfo = devicesList.get(0);
        try {
            handle = MvCameraControl.MV_CC_CreateHandle(stDevInfo);
            if (handle == null) {
                Log.e("CameraControl", "error: CreateHandle fail");
                return;
            }
        } catch (CameraControlException e) {
            e.printStackTrace();
            return;
        }

        int nRet = MvCameraControl.MV_CC_OpenDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: OpenDevice fail" + nRet);
            return;
        }


        nRet = MvCameraControl.MV_CC_SetCommandValue(handle, "DeviceReset");

        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: SetCommandValue fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_CloseDevice(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: CloseDevice fail" + nRet);
            return;
        }

        nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
        if (nRet != MvCameraControlDefines.MV_OK) {
            Log.e("CameraControl", "error: DestroyHandle fail" + nRet);
            return;
        }
    }
}
