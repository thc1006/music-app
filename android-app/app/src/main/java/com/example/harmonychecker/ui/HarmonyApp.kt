package com.example.harmonychecker.ui

import androidx.compose.runtime.Composable
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import com.example.harmonychecker.ui.screens.MainScreen

/**
 * 主應用程式 Composable
 *
 * 負責設定 Navigation 與路由管理。
 */
@Composable
fun HarmonyApp() {
    val navController = rememberNavController()

    NavHost(
        navController = navController,
        startDestination = "main"
    ) {
        // 主畫面
        composable("main") {
            MainScreen(
                onNavigateToCamera = {
                    // TODO: 實作相機導航
                    navController.navigate("camera")
                },
                onNavigateToResult = {
                    // TODO: 實作結果頁導航
                    navController.navigate("result")
                }
            )
        }

        // TODO: 新增其他畫面
        // composable("camera") { CameraScreen(...) }
        // composable("result") { ResultScreen(...) }
    }
}
