package com.example.hk.onecamera.ui;

import android.os.Bundle;
//import android.support.v7.app.AppCompatActivity;
import androidx.appcompat.app.AppCompatActivity;
import android.util.Log;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.LinearLayout;
import android.widget.RelativeLayout;
import android.widget.TextView;
import android.widget.Toast;

import com.example.displayimage.IGlobalLayout;
import com.example.displayimage.MultipleGLSurfaceView;
import com.example.displayimage.PixelFormat;
import com.example.hk.onecamera.CameraManager;
import com.example.hk.onecamera.R;
import com.jaredrummler.materialspinner.MaterialSpinner;

import java.util.ArrayList;
import java.util.List;

import MvCameraControlWrapper.CameraControlException;
import MvCameraControlWrapper.MvCameraControl;
import MvCameraControlWrapper.MvCameraControlDefines;



import static MvCameraControlWrapper.MvCameraControlDefines.MV_OK;
public class MainActivity extends AppCompatActivity implements View.OnClickListener, IMainView {


    EditText inputIntkeyEt;
    EditText inputIntValueEt;


    EditText inputBoolkeyEt;
    EditText inputBoolValueEt;

    EditText inputFloatKeyEt;
    EditText inputFloatValueEt;

    EditText inputStrKeyEt;
    EditText inputStrValueEt;

    EditText inputEnumKeyEt;
    EditText inputEnumValueEt;


    EditText inputEnumByStrKeyEt;
    EditText inputEnumByStrValueEt;

    EditText inputCommandKeyEt;


    private Button enumBt;
    private Button openBt;
    private Button startBt;
    private Button stopBt;
    private Button closeBt;
    private RelativeLayout glViewGroup;
    private MultipleGLSurfaceView multipleGLSurfaceView;
    private MaterialSpinner spinner;
    private TextView logTv;
    private MaterialSpinner setPixelFormatSpinner;
    private ArrayList<MvCameraControlDefines.MV_CC_DEVICE_INFO> deviceList = new ArrayList<>();
    private String[] pixelFormatName = {"MONO8", "YUV422", "RGB"};
    private int[] pixelFormatValue = {PixelFormat.MONO8, PixelFormat.YUV422, PixelFormat.RGB};
    private int selectPixelFormatNum = 0;
    int disPlayPixelFormat = 0;
    int width = 0;
    int height = 0;


    private int selectDeviceNum = 0;
    CameraManager cameraManager;
    GetOneFrameThread getOneFrameThread;
    OpenDeviceThread openDeviceThread;
    CloseDeviceThread closeDeviceThread;
    StartGrabThread startGrabThread;
    StopGrabThread stopGrabThread;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        inputIntkeyEt = findViewById(R.id.inputIntkeyEt);
        inputIntValueEt = findViewById(R.id.inputIntValueEt);

        inputBoolkeyEt = findViewById(R.id.inputBoolkeyEt);
        inputBoolValueEt = findViewById(R.id.inputBoolValueEt);

        inputFloatKeyEt = findViewById(R.id.inputFloatKeyEt);
        inputFloatValueEt = findViewById(R.id.inputFloatValueEt);

        inputStrKeyEt = findViewById(R.id.inputStrKeyEt);
        inputStrValueEt = findViewById(R.id.inputStrValueEt);

        inputEnumKeyEt = findViewById(R.id.inputEnumKeyEt);
        inputEnumValueEt = findViewById(R.id.inputEnumValueEt);

        inputEnumByStrKeyEt = findViewById(R.id.inputEnumByStrKeyEt);
        inputEnumByStrValueEt = findViewById(R.id.inputEnumByStrValueEt);

        inputCommandKeyEt = findViewById(R.id.inputCommandKeyEt);


        enumBt = findViewById(R.id.enumBt);
        glViewGroup = findViewById(R.id.glViewGroup);
        openBt = findViewById(R.id.openBt);
        startBt = findViewById(R.id.startBt);
        stopBt = findViewById(R.id.stopBt);
        closeBt = findViewById(R.id.closeBt);
        spinner = findViewById(R.id.spinner);
        logTv = findViewById(R.id.logTv);
        setPixelFormatSpinner = findViewById(R.id.setPixelFormatSpinner);
        setPixelFormatSpinner.setItems(pixelFormatName);
        setPixelFormatSpinner.setOnItemSelectedListener(new MaterialSpinner.OnItemSelectedListener() {
            @Override
            public void onItemSelected(MaterialSpinner view, int position, long id, Object item) {

                selectPixelFormatNum = position;

            }
        });
        enumBt.setOnClickListener(this);
        openBt.setOnClickListener(this);
        startBt.setOnClickListener(this);
        stopBt.setOnClickListener(this);
        closeBt.setOnClickListener(this);

        findViewById(R.id.setValueBoolBt).setOnClickListener(this);
        findViewById(R.id.getValueBoolBt).setOnClickListener(this);

        findViewById(R.id.setValueIntBt).setOnClickListener(this);
        findViewById(R.id.getValueIntBt).setOnClickListener(this);

        findViewById(R.id.setValueFloatBt).setOnClickListener(this);
        findViewById(R.id.getValueFloatBt).setOnClickListener(this);

        findViewById(R.id.setValueStrBt).setOnClickListener(this);
        findViewById(R.id.getValueStrBt).setOnClickListener(this);

        findViewById(R.id.setValueEnumBt).setOnClickListener(this);
        findViewById(R.id.getValueEnumBt).setOnClickListener(this);

        findViewById(R.id.setValueEnumByStrBt).setOnClickListener(this);

        findViewById(R.id.setValueCommandBt).setOnClickListener(this);

        spinner.setOnItemSelectedListener(new MaterialSpinner.OnItemSelectedListener() {
            @Override
            public void onItemSelected(MaterialSpinner view, int position, long id, Object item) {
                selectDeviceNum = position;
            }
        });
        cameraManager = new CameraManager();
        showLog("SDK版本：" + cameraManager.GetSDKVersion());

        int nTransLayers = MvCameraControl.MV_CC_EnumerateTls();
        if (nTransLayers == MvCameraControlDefines.MV_GIGE_DEVICE) {
            showLog("GigeDevice");
        } else if (nTransLayers == MvCameraControlDefines.MV_USB_DEVICE) {
            showLog("UsbDevice");
        } else if (nTransLayers == (MvCameraControlDefines.MV_GIGE_DEVICE + MvCameraControlDefines.MV_USB_DEVICE)) {
            showLog("GigeDevice and UsbDevice");
        }



    }

    @Override
    protected void onDestroy() {
        super.onDestroy();
        if (closeBt.isEnabled() || stopBt.isEnabled()) {
            if (getOneFrameThread != null) {
                if (getOneFrameThread.isAlive()) {
                    getOneFrameThread.closeflag = true;
                    getOneFrameThread.flag = false;
                } else {
                    int nRet = cameraManager.stopDevice();
                    nRet = cameraManager.closeDevice();
                    cameraManager.destroyHandle();
                    updateButton(ButtonState.CLOSE);
                }
            } else {
                int nRet = cameraManager.stopDevice();
                nRet = cameraManager.closeDevice();
                cameraManager.destroyHandle();
                updateButton(ButtonState.CLOSE);
            }
        }
    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.enumBt: {
                selectDeviceNum = 0;
                deviceList.clear();
                spinner.setItems(new ArrayList());
                try {
                    deviceList = cameraManager.enumDevice();
                } catch (CameraControlException e) {
                    e.printStackTrace();
                    deviceList = null;
                    showLog(e.errMsg + e.errCode);
                    return;
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
                break;
            }

            case R.id.openBt: {
                if (openDeviceThread == null || !openDeviceThread.isAlive()) {
                    openDeviceThread = new OpenDeviceThread();
                    openDeviceThread.start();
                }
                break;
            }
            case R.id.startBt: {


                if (startGrabThread == null) {
                    startGrabThread = new StartGrabThread();
                    startGrabThread.start();
                } else {
                    if (!startGrabThread.isAlive()) {
                        startGrabThread.start();
                    }
                }


                break;
            }
            case R.id.stopBt: {
                if (getOneFrameThread != null) {
                    getOneFrameThread.stopflag = true;
                    getOneFrameThread.flag = false;
                }

                break;
            }
            case R.id.closeBt: {

                if (getOneFrameThread != null) {
                    if (getOneFrameThread.isAlive()) {
                        getOneFrameThread.closeflag = true;
                        getOneFrameThread.flag = false;
                    } else {
                        int nRet = cameraManager.stopDevice();
                        nRet = cameraManager.closeDevice();
                        if (nRet != MvCameraControlDefines.MV_OK) {
                            showLog("closeDevice" + Integer.toHexString(nRet));
                        }
                        cameraManager.destroyHandle();
                        updateButton(ButtonState.CLOSE);
                    }
                } else {
                    int nRet = cameraManager.stopDevice();
                    nRet = cameraManager.closeDevice();
                    if (nRet != MvCameraControlDefines.MV_OK) {
                        // showLog("closeDevice" + Integer.toHexString(nRet));
                    }
                    cameraManager.destroyHandle();
                    updateButton(ButtonState.CLOSE);
                }
                break;
            }

            case R.id.setValueBoolBt: {
                //Boolean
                String key = inputBoolkeyEt.getText().toString();
                String value = inputBoolValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }
                if (value.equals("true") || value.equals("false")) {
                    boolean boolflag = false;
                    if (value.equals("true")) {
                        boolflag = true;
                    }

                    int nRet = cameraManager.setBoolValue(key, boolflag);
                    if (nRet == 0) {
                        showLog("设置成功");
                    } else {
                        showLog("设置失败:" + Integer.toHexString(nRet));
                    }
                } else {
                    showLog("输入的值不对");
                }

                break;
            }
            case R.id.getValueBoolBt: {
                //Bool
                String key = inputBoolkeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key为 空");
                    return;
                }


                Boolean boolValue = Boolean.FALSE;
                int nRet = cameraManager.getBoolValue(key, boolValue);
                if (nRet == 0) {

                    showLog("获取值成功：" + String.valueOf(boolValue.booleanValue()));
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
                break;
            }

            case R.id.setValueIntBt: {
                String key = inputIntkeyEt.getText().toString();
                String value = inputIntValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }
                int nValue = Integer.parseInt(value);

                int nRet = cameraManager.setIntValue(key, nValue);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败:" + Integer.toHexString(nRet));
                }
                break;
            }
            case R.id.getValueIntBt: {
                String key = inputIntkeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key 空");
                    return;
                }
                MvCameraControlDefines.MVCC_INTVALUE intvalue = new MvCameraControlDefines.MVCC_INTVALUE();
                int nRet = cameraManager.getIntValue(key, intvalue);
                if (nRet == 0) {
                    String string = "当前值：" + intvalue.curValue + "--最大值：" + intvalue.max + "--最小值：" + intvalue.min;
                    showLog("获取值成功：" + string);
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }

                Integer intvalue1 = new Integer(0);
                nRet = cameraManager.getIntValue(key, intvalue1);
                if (nRet == 0) {
                    String string = "当前值：" + intvalue1.intValue();
                    showLog("获取值成功：" + string);
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
                break;
            }

            case R.id.setValueFloatBt: {
                String key = inputFloatKeyEt.getText().toString();
                String value = inputFloatValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }
                Float fValue = Float.parseFloat(value);
                int nRet = cameraManager.setFloatValue(key, fValue);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败:" + Integer.toHexString(nRet));
                }
                break;
            }
            case R.id.getValueFloatBt: {
                String key = inputFloatKeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key 空");
                    return;
                }
                //MvCameraControlDefines.MVCC_FLOATVALUE floatvalue = new MvCameraControlDefines.MVCC_FLOATVALUE();
//                int nRet = cameraManager.getFloatValue(key, floatvalue);
//                if (nRet == 0) {
//                    String string = "当前值：" + floatvalue.curValue + "--最大值" + floatvalue.max + "--最小值" + floatvalue.min;
//                    showLog("获取值成功：" + string);
//                } else {
//                    showLog("获取失败：" + Integer.toHexString(nRet));
//                }

                //ChunkModeActive

                Float floatvalue = new Float(0);
                int nRet = cameraManager.getFloatValue(key, floatvalue);
                if (nRet == 0) {
                    String string = "当前值：" + floatvalue.floatValue();
                    showLog("获取值成功：" + string );
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }

/*
                Float floatvalue1 = new Float(0);
                nRet = cameraManager.getFloatValue(key, floatvalue1);
                if (nRet == 0) {
                    String string = "当前值：" + floatvalue1.floatValue();
                    showLog("获取值成功：" + string);
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
*/
                break;
            }
            case R.id.setValueStrBt: {
                String key = inputStrKeyEt.getText().toString();
                String value = inputStrValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }

                int nRet = cameraManager.setStrValu(key, value);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败:" + Integer.toHexString(nRet));
                }
                break;
            }
            case R.id.getValueStrBt: {
                String key = inputStrKeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key 空");
                    return;
                }
                MvCameraControlDefines.MVCC_STRINGVALUE stringvalue = new MvCameraControlDefines.MVCC_STRINGVALUE();
                int nRet = cameraManager.getStrValue(key, stringvalue);
                if (nRet == 0) {
                    String string = "" + stringvalue.curValue;
                    showLog("获取值成功：" + string + "最大长度：" + stringvalue.maxLength + "");
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
                break;
            }
            case R.id.setValueEnumBt: {
                String key = inputEnumKeyEt.getText().toString();
                String value = inputEnumValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }
                int nValue = Integer.parseInt(value);
                int nRet = cameraManager.setEnumValue(key, nValue);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败:" + Integer.toHexString(nRet));
                }
                break;
            }
            case R.id.getValueEnumBt: {
                String key = inputEnumKeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key 空");
                    return;
                }
                MvCameraControlDefines.MVCC_ENUMVALUE enumValue = new MvCameraControlDefines.MVCC_ENUMVALUE();
                int nRet = cameraManager.getEnumValue(key, enumValue);
                if (nRet == 0) {
                    String string = "" + enumValue.curValue + "--";
                    for (int i = 0; i < enumValue.supportValue.size(); i++) {
                        Integer integer = enumValue.supportValue.get(i);
                        string += String.valueOf(integer.intValue()) + "--";
                    }
                    showLog("获取值成功：" + string);
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
/*

                Integer enumValue1 = new Integer(0);
                nRet = cameraManager.getEnumValue(key, enumValue1);
                if (nRet == 0) {
                    String string = "" + enumValue1.intValue() + "--";

                    showLog("获取值成功：" + string);
                } else {
                    showLog("获取失败：" + Integer.toHexString(nRet));
                }
*/

                break;
            }
            case R.id.setValueEnumByStrBt: {
                String key = inputEnumByStrKeyEt.getText().toString();
                String value = inputEnumByStrValueEt.getText().toString();
                if (key.length() == 0 || value.length() == 0) {
                    showLog("key 或 value 为 空");
                    return;
                }
                int nRet = cameraManager.setEnumValueByString(key, value);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败:" + Integer.toHexString(nRet));
                }
                break;
            }

            case R.id.setValueCommandBt: {
                String key = inputCommandKeyEt.getText().toString();
                if (key.length() == 0) {
                    showLog("key 空");
                    return;
                }

                int nRet = cameraManager.setCommandValue(key);
                if (nRet == 0) {
                    showLog("设置成功");
                } else {
                    showLog("设置失败：" + Integer.toHexString(nRet));
                }
                break;
            }
            default: {
                break;
            }

        }
    }

    @Override
    public void updateButton(final int state) {
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                switch (state) {
                    case ButtonState.OPENSUCCESS: {
                        openBt.setEnabled(false);
                        startBt.setEnabled(true);
                        stopBt.setEnabled(false);
                        closeBt.setEnabled(true);
                        break;
                    }
                    case ButtonState.STARtGRAB: {
                        openBt.setEnabled(false);
                        startBt.setEnabled(false);
                        stopBt.setEnabled(true);
                        closeBt.setEnabled(true);
                        break;
                    }
                    case ButtonState.STOPGRAB: {
                        openBt.setEnabled(false);
                        startBt.setEnabled(true);
                        stopBt.setEnabled(false);
                        closeBt.setEnabled(true);
                        break;
                    }
                    case ButtonState.CLOSE: {
                        openBt.setEnabled(true);
                        startBt.setEnabled(false);
                        stopBt.setEnabled(false);
                        closeBt.setEnabled(false);

                        break;
                    }
                }
            }
        });
    }

    @Override
    public void updateImage(byte[] data) {
        multipleGLSurfaceView.updateImage(data);
    }

    @Override
    public void showLog(final String str) {
        Log.e("showLog", str);
        runOnUiThread(new Runnable() {
            @Override
            public void run() {
                if (logTv.getLineCount() > 150) {
                    logTv.setText("");
                    return;
                }
                logTv.append(str + "\n");
                int offset = logTv.getLineCount() * logTv.getLineHeight();
                if (offset > logTv.getHeight()) {
                    logTv.scrollTo(0, offset - logTv.getHeight());
                }
            }
        });

    }


    class GetOneFrameThread extends Thread {
        boolean runFlag = true;
        boolean flag = true;
        boolean stopflag = false;
        boolean closeflag = false;
        byte[] bytes = null;
        MvCameraControlDefines.MV_FRAME_OUT_INFO info = new MvCameraControlDefines.MV_FRAME_OUT_INFO();

        @Override
        public void run() {
            super.run();
            Integer width = new Integer(0);
            Integer height = new Integer(0);

            int nRetW = cameraManager.getIntValue("Width", width);
            int nRetH = cameraManager.getIntValue("Height", height);

            if (nRetW == MV_OK && nRetH == MV_OK) {
                if (bytes == null) {
                    bytes = new byte[width * height * 3];
                } else {
                    if (bytes.length < width * height * 3) {
                        bytes = new byte[width * height * 3];
                    }
                }
            } else {
                showLog("获取图像宽高失败");
                return;
            }


            while (runFlag) {
                if (flag) {
                    int nRet = cameraManager.getOneFrameTimeout(bytes, info, 1000);
                    if (nRet == 0) {

                        updateImage(bytes);
                        Log.d("lcq","bytes = " + bytes);


                        /*
                        String str =
                                "width--------------" + info.width + "\n" +
                                        "height-------------" + info.height + "\n" +
                                        "pixelType----------" + info.pixelType.getnValue() + "\n" +
                                        "frameNum-----------" + info.frameNum + "\n" +
                                        "devTimeStampHigh---" + info.devTimeStampHigh + "\n" +
                                        "ddevTimeStampLow----" + info.devTimeStampLow + "\n" +
                                        "hostTimeStamp------" + info.hostTimeStamp + "\n" +
                                        "frameLen-----------" + info.frameLen + "\n" +
                                        "secondCount--------" + info.secondCount + "\n" +
                                        "cycleCount---------" + info.cycleCount + "\n" +
                                        "cycleOffset--------" + info.cycleOffset + "\n" +
                                        "gain---------------" + info.gain + "\n" +
                                        "exposureTime-------" + info.exposureTime + "\n" +
                                        "averageBrightness--" + info.averageBrightness + "\n" +
                                        "red----------------" + info.red + "\n" +
                                        "green--------------" + info.green + "\n" +
                                        "blue---------------" + info.blue + "\n" +
                                        "frameCounter-------" + info.frameCounter + "\n" +
                                        "triggerIndex-------" + info.triggerIndex + "\n" +
                                        "input--------------" + info.input + "\n" +
                                        "output-------------" + info.output + "\n" +
                                        "offsetX------------" + info.offsetX + "\n" +
                                        "offsetY------------" + info.offsetY + "\n" +
                                        "chunkWidth---------" + info.chunkWidth + "\n" +
                                        "chunkHeight--------" + info.chunkHeight + "\n" +
                                        "lostPacket---------" + info.lostPacket;




*/

                        //   showLog(str);

                    }

                } else {
                    if (stopflag) {
                        stopflag = false;
                        updateButton(ButtonState.STOPGRAB);
                        Log.e("StopGrabThread", "StopGrabThread");
                        int nRet = cameraManager.stopDevice();
                        if (nRet != MvCameraControlDefines.MV_OK) {
                            showLog("stopDevice" + Integer.toHexString(nRet));
                        }
                    } else if (closeflag) {
                        closeflag = false;
                        int nRet = cameraManager.stopDevice();
                        nRet = cameraManager.closeDevice();
                        cameraManager.destroyHandle();
                        updateButton(ButtonState.CLOSE);

                        runFlag = false;
                    }
                }
            }
        }
    }


    class OpenDeviceThread extends Thread {
        @Override
        public void run() {

            if (deviceList.size() == 0) {
                showLog("清先枚举设备");
                return;
            }

            try {
                cameraManager.createHandle(deviceList.get(selectDeviceNum));
            } catch (CameraControlException e) {
                e.printStackTrace();
                showLog(e.errMsg + e.errCode);
                return;
            }

            int nRet = cameraManager.openDevice();

            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("OpenDevice fail nRet = " + Integer.toHexString(nRet));
                return;
            }


            updateButton(ButtonState.OPENSUCCESS);
        }
    }

    class CloseDeviceThread extends Thread {
        @Override
        public void run() {
            super.run();
            int nRet = cameraManager.stopDevice();
            nRet = cameraManager.closeDevice();
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("closeDevice" + Integer.toHexString(nRet));
            }
            cameraManager.destroyHandle();
            updateButton(ButtonState.CLOSE);
        }
    }

    class StartGrabThread extends Thread {
        @Override
        public void run() {
            super.run();


            if (deviceList.get(selectDeviceNum).transportLayerType == MvCameraControlDefines.MV_GIGE_DEVICE) {


                int nPacketSize = cameraManager.getOptimalPacketSize();
                if (nPacketSize > 0) {
                    int nRet = cameraManager.setIntValue("GevSCPSPacketSize", nPacketSize);
                    if (nRet != MV_OK) {
                        showLog("Warning: Set Packet Size fail nRet" + Integer.toHexString(nRet));
                    } else {
                        showLog("set GevSCPSPacketSize 1500");
                    }
                } else {
                    showLog("Warning: Get Packet Size fail nRet" + Integer.toHexString(nPacketSize));
                }

                Integer GevSCPD = new Integer(0);
                int nRet = cameraManager.getIntValue("GevSCPD", GevSCPD);
                if (nRet != MV_OK) {
                    showLog("get GevSCPD fail nRet = " + Integer.toHexString(nRet));
                } else {
                    showLog("GevSCPD = " + GevSCPD.intValue());
                }

                nRet = cameraManager.setIntValue("GevSCPD", 2000);
                if (nRet == MV_OK) {
                    showLog("Set GevSCPD success");
                } else {
                    showLog("Set GevSCPD fail");
                }

                nRet = cameraManager.getIntValue("GevSCPD", GevSCPD);
                if (nRet != MV_OK) {
                    showLog("get GevSCPD fail nRet = " + Integer.toHexString(nRet));
                } else {
                    showLog("GevSCPD = " + GevSCPD.intValue());
                }
            }


            final Integer width = new Integer(0);
            int nRet = cameraManager.getIntValue("Width", width);
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("get Width fail nRet = " + Integer.toHexString(nRet));
                return;
            }

            final Integer height = new Integer(0);
            nRet = cameraManager.getIntValue("Height", height);
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("get Height fail nRet = " + Integer.toHexString(nRet));
                return;
            }

            if (selectPixelFormatNum >= pixelFormatValue.length) {
                selectPixelFormatNum = 0;
            }
//
//            nRet = cameraManager.setEnumValue("PixelFormat", pixelFormatValue[selectPixelFormatNum]);
//            if (nRet != MvCameraControlDefines.MV_OK) {
//                showLog("set PixelFormat fail nRet = " + Integer.toHexString(nRet));
//            }


            final Integer pixelFormat = new Integer(0);
            nRet = cameraManager.getEnumValue("PixelFormat", pixelFormat);
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("get PixelFormat fail nRet = " + Integer.toHexString(nRet));
                return;
            }


            runOnUiThread(new Runnable() {
                @Override
                public void run() {
                    updatePixelFormatView(pixelFormat, width, height);
                }
            });




            nRet = cameraManager.startDevice();
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("startDevice" + Integer.toHexString(nRet));
            }



            updateButton(ButtonState.STARtGRAB);


            if (getOneFrameThread == null) {
                getOneFrameThread = new GetOneFrameThread();
                getOneFrameThread.flag = true;
                getOneFrameThread.runFlag = true;
                getOneFrameThread.start();
            } else {
                getOneFrameThread.flag = true;
                getOneFrameThread.runFlag = true;
                getOneFrameThread.start();
            }
        }
    }

    class StopGrabThread extends Thread {
        @Override
        public void run() {
            super.run();
            updateButton(ButtonState.STOPGRAB);
            Log.e("StopGrabThread", "StopGrabThread");
            int nRet = cameraManager.stopDevice();
            if (nRet != MvCameraControlDefines.MV_OK) {
                showLog("stopDevice" + Integer.toHexString(nRet));
            }
        }
    }


    private long mPressedTime = 0;

    @Override
    public void onBackPressed() {
        long mNowTime = System.currentTimeMillis();//获取第一次按键时间
        if ((mNowTime - mPressedTime) > 500) {//比较两次按键时间差
            Toast.makeText(this, "再按一次退出程序", Toast.LENGTH_SHORT).show();
            mPressedTime = mNowTime;
        } else {//退出程序
            this.finish();
            System.exit(0);
        }
    }


    private void updatePixelFormatView(final Integer pixelFormat, final Integer w, final Integer h) {
        if (disPlayPixelFormat != pixelFormat || w != width || h != height) {
            disPlayPixelFormat = pixelFormat;
            width = w;
            height = h;
            glViewGroup.removeAllViews();
            multipleGLSurfaceView = new MultipleGLSurfaceView(MainActivity.this, new IGlobalLayout() {
                @Override
                public void lodfinish() {
                    multipleGLSurfaceView.pixelFormat(35127316);
                    multipleGLSurfaceView.setSize(w.intValue(), h.intValue());
                }
            });
            LinearLayout.LayoutParams params = new LinearLayout.LayoutParams(
                    LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.MATCH_PARENT);
            multipleGLSurfaceView.setLayoutParams(params);
            glViewGroup.addView(multipleGLSurfaceView);

        }
    }

}
