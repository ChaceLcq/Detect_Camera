package com.example.hk.onecamera.ui;

import java.util.UUID;

/**
 * Created by panfeilong on 2019/11/2.
 */

public class UUIDUtils  {
    public static ClassLoader getClassLoader() {
        return UUIDUtils.class.getClassLoader();
    }
    public static String get(){
        return UUID.randomUUID().toString();
    }

}
