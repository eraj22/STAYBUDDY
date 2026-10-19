import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';

import '../theme.dart';

/// A compact identity marker for top-level Owner tabs.
///
/// Primary navigation is handled by the Owner bottom bar, so these pages
/// should not expose a back action that can accidentally close the workspace.
class OwnerPortalBadge extends StatelessWidget {
  final bool darkSurface;

  const OwnerPortalBadge({super.key, this.darkSurface = true});

  @override
  Widget build(BuildContext context) {
    final foreground = darkSurface ? Colors.white : AppTheme.tealDeep;

    return Semantics(
      label: 'Owner Portal',
      container: true,
      child: Container(
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 9),
        decoration: BoxDecoration(
          color: darkSurface
              ? Colors.white.withValues(alpha: 0.13)
              : AppTheme.teal.withValues(alpha: 0.08),
          borderRadius: BorderRadius.circular(22),
          border: Border.all(
            color: darkSurface
                ? Colors.white.withValues(alpha: 0.18)
                : AppTheme.teal.withValues(alpha: 0.16),
          ),
        ),
        child: Row(
          mainAxisSize: MainAxisSize.min,
          children: [
            Icon(
              Icons.admin_panel_settings_rounded,
              size: 15,
              color: darkSurface ? AppTheme.tealMint : AppTheme.teal,
            ),
            const SizedBox(width: 7),
            Text(
              'Owner Portal',
              style: GoogleFonts.dmSans(
                color: foreground,
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
                letterSpacing: 0.1,
              ),
            ),
          ],
        ),
      ),
    );
  }
}
