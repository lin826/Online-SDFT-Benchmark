package ai.onlinesdft.router.ui

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.Immutable
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * Product palette for the notification assistant. Route colours are the only
 * hues that carry meaning; everything else is a neutral so the three decisions
 * stay legible at a glance.
 */
@Immutable
internal data class RouterColors(
    val background: Color,
    val surface: Color,
    val surfaceMuted: Color,
    val surfaceSunken: Color,
    val ink: Color,
    val inkMuted: Color,
    val inkFaint: Color,
    val outline: Color,
    val brand: Color,
    val brandSoft: Color,
    val onBrand: Color,
    val now: Color,
    val later: Color,
    val silent: Color,
    val warning: Color,
    val warningSoft: Color,
    val track: Color,
)

private val LightColors = RouterColors(
    background = Color(0xFFF4F6F8),
    surface = Color(0xFFFFFFFF),
    surfaceMuted = Color(0xFFEFF2F5),
    surfaceSunken = Color(0xFFF8FAFB),
    ink = Color(0xFF0F1419),
    inkMuted = Color(0xFF59636E),
    inkFaint = Color(0xFF8A939E),
    outline = Color(0xFFE3E8EE),
    brand = Color(0xFF1F6F4A),
    brandSoft = Color(0xFFE6F2EC),
    onBrand = Color(0xFFFFFFFF),
    now = Color(0xFFB4530A),
    later = Color(0xFF2050C8),
    silent = Color(0xFF6B7684),
    warning = Color(0xFFB4530A),
    warningSoft = Color(0xFFFCF2E4),
    track = Color(0xFFE7EBF0),
)

private val DarkColors = RouterColors(
    background = Color(0xFF0D1014),
    surface = Color(0xFF161B21),
    surfaceMuted = Color(0xFF1D242C),
    surfaceSunken = Color(0xFF12171C),
    ink = Color(0xFFE9EDF2),
    inkMuted = Color(0xFF9BA6B2),
    inkFaint = Color(0xFF6F7A87),
    outline = Color(0xFF262E37),
    brand = Color(0xFF5FBF8F),
    brandSoft = Color(0xFF16281F),
    onBrand = Color(0xFF06120C),
    now = Color(0xFFE9A15C),
    later = Color(0xFF7BA6F5),
    silent = Color(0xFF98A3B0),
    warning = Color(0xFFE9A15C),
    warningSoft = Color(0xFF241C11),
    track = Color(0xFF232B34),
)

internal val LocalRouterColors = staticCompositionLocalOf { LightColors }

internal val routerColors: RouterColors
    @Composable get() = LocalRouterColors.current

@Composable
internal fun RouterTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    content: @Composable () -> Unit,
) {
    val colors = if (darkTheme) DarkColors else LightColors
    val scheme = if (darkTheme) {
        darkColorScheme(
            primary = colors.brand,
            onPrimary = colors.onBrand,
            secondary = colors.later,
            tertiary = colors.now,
            background = colors.background,
            onBackground = colors.ink,
            surface = colors.surface,
            onSurface = colors.ink,
            surfaceVariant = colors.surfaceMuted,
            onSurfaceVariant = colors.inkMuted,
            outline = colors.outline,
        )
    } else {
        lightColorScheme(
            primary = colors.brand,
            onPrimary = colors.onBrand,
            secondary = colors.later,
            tertiary = colors.now,
            background = colors.background,
            onBackground = colors.ink,
            surface = colors.surface,
            onSurface = colors.ink,
            surfaceVariant = colors.surfaceMuted,
            onSurfaceVariant = colors.inkMuted,
            outline = colors.outline,
        )
    }
    val base = MaterialTheme.typography
    CompositionLocalProvider(LocalRouterColors provides colors) {
        MaterialTheme(
            colorScheme = scheme,
            typography = Typography(
                displaySmall = base.displaySmall.copy(
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = (-1.2).sp,
                ),
                headlineSmall = base.headlineSmall.copy(
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = (-0.5).sp,
                ),
                titleLarge = base.titleLarge.copy(
                    fontWeight = FontWeight.SemiBold,
                    letterSpacing = (-0.3).sp,
                ),
                titleMedium = base.titleMedium.copy(fontWeight = FontWeight.SemiBold),
                titleSmall = base.titleSmall.copy(fontWeight = FontWeight.SemiBold),
                labelLarge = base.labelLarge.copy(fontWeight = FontWeight.Medium),
            ),
            content = content,
        )
    }
}
