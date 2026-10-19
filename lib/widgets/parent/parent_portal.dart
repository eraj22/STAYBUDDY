import 'package:flutter/material.dart';

import '../responsive_shell.dart';

/// Keeps every parent route and its overlays inside the same phone viewport.
class ParentPortal extends StatelessWidget {
  const ParentPortal({
    super.key,
    required this.navigatorKey,
    required this.builder,
  });

  final GlobalKey<NavigatorState> navigatorKey;
  final WidgetBuilder builder;

  @override
  Widget build(BuildContext context) {
    return ResponsiveShell(
      child: LayoutBuilder(
        builder: (context, constraints) {
          return MediaQuery(
            data: MediaQuery.of(context).copyWith(size: constraints.biggest),
            child: NavigatorPopHandler<Object?>(
              onPopWithResult: (result) {
                navigatorKey.currentState?.pop(result);
              },
              child: Navigator(
                key: navigatorKey,
                onGenerateRoute: (settings) => MaterialPageRoute<void>(
                  settings: settings,
                  builder: builder,
                ),
              ),
            ),
          );
        },
      ),
    );
  }
}
