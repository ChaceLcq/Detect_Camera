package com.example.hk.onecamera.ui;

/**
 * Created by panfeilong on 2019/11/1.
 */

public interface IMainView {
    void showLog(String str);
    void updateButton(final int state);
    void updateImage(byte[] data);
}
