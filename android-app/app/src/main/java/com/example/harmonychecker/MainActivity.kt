package com.example.harmonychecker

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import com.example.harmonychecker.ui.HarmonyApp
import com.example.harmonychecker.ui.theme.HarmonyCheckerTheme

/**
 * 主 Activity
 *
 * 使用 Jetpack Compose 建立 UI。
 * 本 Activity 僅負責設定主題與啟動 Compose UI tree。
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        setContent {
            HarmonyCheckerTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    HarmonyApp()
                }
            }
        }
    }
}
