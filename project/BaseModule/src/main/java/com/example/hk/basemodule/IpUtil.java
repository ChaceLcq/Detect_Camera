package com.example.hk.basemodule;

import android.content.Context;

import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Created by panfeilong on 2019/9/10.
 */

public class IpUtil {
    /**
     * =========通过ip ping 来判断ip是否通
     *
     * @param ip
     */
    public static boolean judgeTheConnect(Context context, String ip) {

        try {

            if (ip != null) {

                //代表ping 3 次 超时时间为10秒
                Process p = Runtime.getRuntime().exec("ping -c 2 -w 10 " + ip);//ping3次

                int status = p.waitFor();

                if (status == 0) {
                    //代表成功
//                    Toast.makeText(context, "成功", Toast.LENGTH_SHORT).show();
                    return true;
                } else {
                    //代表失败
//                    Toast.makeText(context, "失败", Toast.LENGTH_SHORT).show();
                    return false;
                }
            } else {
                //代表失败
//                Toast.makeText(context, "IP为空", Toast.LENGTH_SHORT).show();
                return false;
            }
        } catch (Exception e) {
          //  Log.e("异常", e.getMessage());
            return false;
        }

    }

    /**
     * 判断是否为合法IP * @return the ip
     */
    public static boolean isboolIp(String ipAddress) {
        String ip = "([1-9]|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])(\\.(\\d|[1-9]\\d|1\\d{2}|2[0-4]\\d|25[0-5])){3}";
        Pattern pattern = Pattern.compile(ip);
        Matcher matcher = pattern.matcher(ipAddress);
        return matcher.matches();
    }


}
