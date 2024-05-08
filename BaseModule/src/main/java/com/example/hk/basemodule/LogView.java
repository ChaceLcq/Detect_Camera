package com.example.hk.basemodule;

import android.app.Activity;
import android.view.View;
import android.widget.EditText;

public class LogView {
    public static void showLog(Activity activity, final View view, final String str) {
        activity.runOnUiThread(new Runnable() {
            @Override
            public void run() {
                EditText editText = (EditText) view;
                if (editText.getLineCount() > 150) {
                    editText.setText("");
                    return;
                }
                editText.append(str + "\n");
                int offset = editText.getLineCount() * editText.getLineHeight();
                if (offset > editText.getHeight()) {
                    editText.scrollTo(0, offset - editText.getHeight());
                }
            }
        });
    }
}
