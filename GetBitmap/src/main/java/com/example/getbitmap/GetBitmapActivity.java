package com.example.getbitmap;

import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.support.v7.app.AppCompatActivity;
import android.text.method.ScrollingMovementMethod;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.ImageView;
import android.widget.Toast;
import android.util.Log;

import com.example.hk.basemodule.IMainView;
import com.example.hk.basemodule.LogView;
import com.jaredrummler.materialspinner.MaterialSpinner;

import java.util.ArrayList;
import java.util.List;

import MvCameraControlWrapper.CameraControlException;
import MvCameraControlWrapper.MvCameraControl;
import MvCameraControlWrapper.MvCameraControlDefines;

//import org.opencv.*;

import static MvCameraControlWrapper.MvCameraControlDefines.MV_OK;

public class GetBitmapActivity extends AppCompatActivity implements IMainView {

    private ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> deviceList = new ArrayList<>();
    private MaterialSpinner spinner;
    private ImageView imageView;
    EditText logTv;
    int selectNum = 0;
    Button enumDevicesBt;
    Button closeDeviceBt;
    Button openDeviceBt;
    byte[] datas = null;
    MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();
    GetBitmapThread getBitmapThread;
    MvCameraControlDefines.Handle handle = null;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_get_bitmap);
        initView();
        initListener();
    }

    private void initView() {
        spinner = findViewById(R.id.spinner);
        logTv = findViewById(R.id.logTv);
        logTv.setMovementMethod(ScrollingMovementMethod.getInstance());
        enumDevicesBt = findViewById(R.id.enumDevicesBt);
        openDeviceBt = findViewById(R.id.openDeviceBt);
        closeDeviceBt = findViewById(R.id.closeDeviceBt);
        imageView = findViewById(R.id.imageView);
        spinner.setOnItemSelectedListener(new MaterialSpinner.OnItemSelectedListener() {
            @Override
            public void onItemSelected(MaterialSpinner view, int position, long id, Object item) {
                selectNum = position;
            }
        });
    }

    private void initListener() {
        findViewById(R.id.enumDevicesBt).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                selectNum = 0;
                deviceList.clear();
                spinner.setItems(new ArrayList());
                try {
                    deviceList = MvCameraControl.MV_CC_EnumDevices(MvCameraControlDefines.MV_USB_DEVICE | MvCameraControlDefines.MV_GIGE_DEVICE);
                } catch (CameraControlException e) {
                    e.printStackTrace();
                }
                if (deviceList != null) {
                    int size = deviceList.size();
                    if (size > 0) {
                        List<String> deviceName = new ArrayList<>();
                        for (int i = 0; i < size; i++) {
                            MvCameraControlDefines.MV_CC_DEVICE_INFO entity = deviceList.get(i);
                            if (entity.transportLayerType == MvCameraControlDefines.MV_GIGE_DEVICE) {
                                String str = "[" + String.valueOf(i) + "]"
                                        + entity.gigEInfo.manufacturerName + "--"
                                        + entity.gigEInfo.serialNumber + "--"
                                        + entity.gigEInfo.deviceVersion + "--"
                                        + entity.gigEInfo.userDefinedName;
                                deviceName.add(str);
                            } else {
                                String str = "[" + String.valueOf(i) + "]"
                                        + entity.usb3VInfo.manufacturerName + "--"
                                        + entity.usb3VInfo.serialNumber + "--"
                                        + entity.usb3VInfo.deviceVersion + "--"
                                        + entity.usb3VInfo.userDefinedName;
                                deviceName.add(str);
                            }

                        }
                        spinner.setItems(deviceName);
                    } else {
                        spinner.setItems(new ArrayList());
                    }
                } else {
                    showLog("未枚举到设备");
                }
            }
        });

        findViewById(R.id.openDeviceBt).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                openDevice();
            }
        });

        findViewById(R.id.closeDeviceBt).setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                closeDevice();
            }
        });
    }

    private void openDevice() {
        if (getBitmapThread == null) {
            getBitmapThread = new GetBitmapThread(new IExitGetOneFrame() {
                @Override
                public void exit() {
                    showLog("正在关闭...");
                    // 停止取流
                    // end grab image
                    int nRet = MvCameraControl.MV_CC_StopGrabbing(handle);
                    if (MV_OK != nRet) {
                        showLog("MV_CC_StopGrabbing fail! nRet" + Integer.toHexString(nRet));
                    }
                    // 关闭设备
                    // close device
                    nRet = MvCameraControl.MV_CC_CloseDevice(handle);
                    if (MV_OK != nRet) {
                        showLog("MV_CC_CloseDevice fail! nRet " + nRet);
                    }
                    // 销毁句柄
                    // destroy handle
                    nRet = MvCameraControl.MV_CC_DestroyHandle(handle);
                    if (MV_OK != nRet) {
                        showLog("MV_CC_DestroyHandle fail! nRet" + nRet);
                    }
                    showLog("关闭完毕...");
                }
            });
            getBitmapThread.flag = true;
            getBitmapThread.start();
        } else {
            if (!getBitmapThread.isAlive()) {
                getBitmapThread.flag = true;
                getBitmapThread.start();
            } else {
                showLog("请先关闭设备");
            }
        }
    }

    @Override
    public void showLog(final String str) {
        LogView.showLog(this, logTv, str);

    }

    @Override
    public void updateButton(int state) {

    }

    @Override
    public void updateImage(byte[] data) {

    }

    public void toast(final String string) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                Toast.makeText(GetBitmapActivity.this, string, Toast.LENGTH_SHORT).show();
            }
        });
    }


    private void closeDevice() {

        if (getBitmapThread != null) {
            getBitmapThread.flag = false;
        }

    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (getBitmapThread != null) {
            if (getBitmapThread.flag) {
                closeDevice();
            }
        }
    }

    class GetBitmapThread extends Thread {

        boolean flag = true;
        IExitGetOneFrame iExitGetOneFrame;

        GetBitmapThread(IExitGetOneFrame iExitGetOneFrame) {
            this.iExitGetOneFrame = iExitGetOneFrame;
        }


        @Override
        public void run() {
            super.run();

            try {
                setEnumBt(false);
                setCloseDevBt(false);
                showLog("正在打开...");
                if (deviceList != null && deviceList.size() > 0) {
                    try {
                        handle = MvCameraControl.MV_CC_CreateHandle(deviceList.get(selectNum));
                    } catch (CameraControlException e) {
                        e.printStackTrace();
                    }

                    if (handle == null) {
                        showLog("CreateHandle fail! nRet");
                        return;
                    }
                    int nRet = MV_OK;
                    nRet = MvCameraControl.MV_CC_OpenDevice(handle);
                    if (nRet != 0) {
                        showLog("OpenDevice fail! nRet " + Integer.toHexString(nRet));
                        return;
                    }
                    if (deviceList.get(selectNum).transportLayerType == MvCameraControlDefines.MV_GIGE_DEVICE) {
                        int nPacketSize = MvCameraControl.MV_CC_GetOptimalPacketSize(handle);
                        showLog("nPacketSize:" + nPacketSize + "");
                        if (nPacketSize > 0) {
                            nRet = MvCameraControl.MV_CC_SetIntValue(handle, "GevSCPSPacketSize", nPacketSize);
                            if (nRet != MV_OK) {
                                showLog("Warning: Set Packet Size fail nRet" + Integer.toHexString(nRet));
                                return;
                            }
                        } else {
                            showLog("Warning: Get Packet Size fail nRet" + Integer.toHexString(nPacketSize));
                        }
                    }

                    nRet = MvCameraControl.MV_CC_SetFloatValue(handle, "AcquisitionFrameRate", 2);
                    if (MV_OK != nRet) {
                        showLog("AcquisitionFrameRate fail! nRet " + Integer.toHexString(nRet));
                    }
                    nRet = MvCameraControl.MV_CC_SetBoolValue(handle, "AcquisitionFrameRateEnable", true);
                    if (MV_OK != nRet) {
                        showLog("AcquisitionFrameRateEnable fail! nRet " + Integer.toHexString(nRet));
                    }

                    nRet = MvCameraControl.MV_CC_SetEnumValue(handle, "TriggerMode", 0);
                    if (MV_OK != nRet) {
                        showLog("MV_CC_SetTriggerMode fail! nRet " + Integer.toHexString(nRet));
                        return;
                    }

                    Integer width = new Integer(0);
                    nRet = MvCameraControl.MV_CC_GetIntValue(handle, "Width", width);
                    if (nRet != MV_OK) {
                        showLog("GetWidth fail! nRet " + Integer.toHexString(nRet));
                        return;
                    }
                    Integer height = new Integer(0);
                    nRet = MvCameraControl.MV_CC_GetIntValue(handle, "Height", height);
                    if (nRet != MV_OK) {
                        showLog("GetHeight fail! nRet " + Integer.toHexString(nRet));
                        return;
                    }
                    datas = new byte[width * height * 4];

                    // 开始取流
                    // start grab image
                    nRet = MvCameraControl.MV_CC_StartGrabbing(handle);
                    if (MV_OK != nRet) {
                        showLog("MV_CC_StartGrabbing fail! nRet" + nRet);
                        return;
                    }
                } else {
                    toast("请先枚举设备");
                    showLog("打开完毕...");
                    return;
                }
            } finally {
                setEnumBt(true);
                setCloseDevBt(true);
            }

            while (true) {
                if (!flag) {
                    if (iExitGetOneFrame != null) {
                        iExitGetOneFrame.exit();
                    }
                    return;
                } else {
                    int nRet1 = MvCameraControl.MV_CC_GetBitmapTimeout(handle, datas, info, 2000);
                    if (nRet1 == MV_OK) {
                        showLog("GetOneFrame,Width:" + info.width +
                                "Height:" + info.height +
                                "nFrameNum" + info.frameNum);
                        runOnUiThread(new Runnable() {
                            @Override
                            public void run() {
                                BitmapFactory.Options options = new BitmapFactory.Options();
                                options.inSampleSize = 2;
                                Bitmap bm = BitmapFactory.decodeByteArray(datas, 0, info.frameLen, options);
                                Log.d("lcq","bm = " + bm);
                                if (bm != null) {
                                    imageView.setImageBitmap(bm);
                                }
                            }
                        });
                    } else {
                        showLog("OneFrameTimeout fail：nRet = " + Integer.toHexString(nRet1));
                        try {
                            Thread.sleep(1000);
                        } catch (InterruptedException e) {
                            e.printStackTrace();
                        }
                    }
                }
            }
        }
    }


    public void setEnumBt(final boolean flag) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                enumDevicesBt.setEnabled(flag);
            }
        });
    }

    public void setCloseDevBt(final boolean flag) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                closeDeviceBt.setEnabled(flag);
            }
        });
    }
}
