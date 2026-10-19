import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentNotificationsScreen extends StatefulWidget {
  const ParentNotificationsScreen({super.key});

  @override
  State<ParentNotificationsScreen> createState() =>
      _ParentNotificationsScreenState();
}

class _ParentNotificationsScreenState
    extends State<ParentNotificationsScreen> {
  late VideoPlayerController _videoController;

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);
  static const Color red = Color(0xFFFF6B6B);
  static const Color warning = Color(0xFFFFD166);
  static const Color purple = Color(0xFFB7A8FF);

  String _selectedFilter = 'All';

  final List<String> _filters = const [
    'All',
    'Attendance',
    'Fees',
    'Emergency',
    'General',
  ];

  final List<_ParentNotification> _notifications = [
    _ParentNotification(
      title: 'Ali entered the hostel',
      subtitle: 'Your child entered through the main gate.',
      time: 'Today • 7:45 AM',
      type: 'Attendance',
      icon: Icons.login_rounded,
      color: green,
      unread: true,
    ),
    _ParentNotification(
      title: 'Emergency alert received',
      subtitle:
          'Ali triggered the emergency button. Hostel administration was also notified.',
      time: 'Today • 4:25 PM',
      type: 'Emergency',
      icon: Icons.emergency_outlined,
      color: red,
      unread: true,
    ),
    _ParentNotification(
      title: 'Fee reminder',
      subtitle:
          'August hostel fee is due in 5 days.',
      time: 'Today • 11:15 AM',
      type: 'Fees',
      icon: Icons.payments_outlined,
      color: warning,
      unread: true,
    ),
    _ParentNotification(
      title: 'Location sharing active',
      subtitle:
          'Ali\'s location was updated recently.',
      time: 'Today • 10:42 AM',
      type: 'General',
      icon: Icons.location_on_outlined,
      color: mint,
      unread: false,
    ),
    _ParentNotification(
      title: 'New hostel announcement',
      subtitle:
          'Dinner timing has been updated to 7:30 PM - 9:30 PM.',
      time: 'Yesterday • 6:30 PM',
      type: 'General',
      icon: Icons.campaign_outlined,
      color: purple,
      unread: false,
    ),
    _ParentNotification(
      title: 'Ali left the hostel',
      subtitle:
          'Your child exited through the main gate.',
      time: 'Yesterday • 5:55 PM',
      type: 'Attendance',
      icon: Icons.logout_rounded,
      color: red,
      unread: false,
    ),
  ];

  @override
  void initState() {
    super.initState();

    _videoController = VideoPlayerController.asset(
      'assets/images/background.mp4',
    )
      ..initialize().then((_) {
        if (!mounted) return;

        _videoController
          ..setLooping(true)
          ..setVolume(0)
          ..play();

        setState(() {});
      });
  }

  @override
  void dispose() {
    _videoController.dispose();
    super.dispose();
  }

  List<_ParentNotification> get _filteredNotifications {
    if (_selectedFilter == 'All') {
      return _notifications;
    }

    return _notifications
        .where(
          (item) => item.type == _selectedFilter,
        )
        .toList();
  }

  void _markAllRead() {
    setState(() {
      for (final item in _notifications) {
        item.unread = false;
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: background,
      body: Stack(
        children: [
          _buildBackground(),

          Positioned.fill(
            child: Container(
              color: Colors.black.withOpacity(.46),
            ),
          ),

          SafeArea(
            child: Column(
              children: [
                _buildTopBar(),

                Expanded(
                  child: SingleChildScrollView(
                    physics: const BouncingScrollPhysics(),
                    padding: const EdgeInsets.fromLTRB(
                      18,
                      10,
                      18,
                      30,
                    ),
                    child: Center(
                      child: ConstrainedBox(
                        constraints: const BoxConstraints(
                          maxWidth: 600,
                        ),
                        child: Column(
                          crossAxisAlignment:
                              CrossAxisAlignment.start,
                          children: [
                            _buildSummaryCard(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Filter Notifications',
                              'View alerts by category',
                            ),

                            const SizedBox(height: 12),

                            _buildFilters(),

                            const SizedBox(height: 22),

                            _sectionTitle(
                              'Recent Alerts',
                              'Latest updates related to your child',
                            ),

                            const SizedBox(height: 12),

                            if (_filteredNotifications.isEmpty)
                              _buildEmptyState()
                            else
                              ..._filteredNotifications.map(
                                (item) => Padding(
                                  padding:
                                      const EdgeInsets.only(
                                    bottom: 11,
                                  ),
                                  child:
                                      _notificationCard(item),
                                ),
                              ),
                          ],
                        ),
                      ),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildBackground() {
    if (!_videoController.value.isInitialized) {
      return const Positioned.fill(
        child: ColoredBox(
          color: background,
        ),
      );
    }

    return Positioned.fill(
      child: FittedBox(
        fit: BoxFit.cover,
        child: SizedBox(
          width: _videoController.value.size.width,
          height:
              _videoController.value.size.height,
          child: VideoPlayer(
            _videoController,
          ),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        14,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () =>
                Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),

          const SizedBox(width: 4),

          Expanded(
            child: Text(
              'Notifications',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          TextButton(
            onPressed: _markAllRead,
            child: Text(
              'Mark all read',
              style: GoogleFonts.dmSans(
                color: mint,
                fontSize: 10.5,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildSummaryCard() {
    final unreadCount = _notifications
        .where((item) => item.unread)
        .length;

    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(19),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E)
                .withOpacity(.96),
            const Color(0xFF003B3B)
                .withOpacity(.96),
          ],
        ),
        borderRadius:
            BorderRadius.circular(27),
        border: Border.all(
          color: mint.withOpacity(.20),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: teal.withOpacity(.14),
              borderRadius:
                  BorderRadius.circular(18),
            ),
            child: const Icon(
              Icons.notifications_active_outlined,
              color: mint,
              size: 29,
            ),
          ),

          const SizedBox(width: 14),

          Expanded(
            child: Column(
              crossAxisAlignment:
                  CrossAxisAlignment.start,
              children: [
                Text(
                  '$unreadCount Unread Notifications',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 16,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  'Attendance, fees, emergency and hostel updates appear here.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white60,
                    fontSize: 10.8,
                    height: 1.4,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _sectionTitle(
    String title,
    String subtitle,
  ) {
    return Column(
      crossAxisAlignment:
          CrossAxisAlignment.start,
      children: [
        Text(
          title,
          style: GoogleFonts.dmSans(
            color: Colors.white,
            fontSize: 18,
            fontWeight: FontWeight.w800,
          ),
        ),

        const SizedBox(height: 2),

        Text(
          subtitle,
          style: GoogleFonts.dmSans(
            color: Colors.white38,
            fontSize: 11,
          ),
        ),
      ],
    );
  }

  Widget _buildFilters() {
    return SingleChildScrollView(
      scrollDirection: Axis.horizontal,
      physics: const BouncingScrollPhysics(),
      child: Row(
        children: _filters.map((filter) {
          final selected =
              filter == _selectedFilter;

          return Padding(
            padding:
                const EdgeInsets.only(right: 8),
            child: InkWell(
              onTap: () {
                setState(() {
                  _selectedFilter = filter;
                });
              },
              borderRadius:
                  BorderRadius.circular(30),
              child: AnimatedContainer(
                duration: const Duration(
                  milliseconds: 180,
                ),
                padding:
                    const EdgeInsets.symmetric(
                  horizontal: 14,
                  vertical: 9,
                ),
                decoration: BoxDecoration(
                  color: selected
                      ? teal.withOpacity(.18)
                      : darkTeal.withOpacity(.67),
                  borderRadius:
                      BorderRadius.circular(30),
                  border: Border.all(
                    color: selected
                        ? mint.withOpacity(.45)
                        : mint.withOpacity(.09),
                  ),
                ),
                child: Text(
                  filter,
                  style: GoogleFonts.dmSans(
                    color: selected
                        ? mint
                        : Colors.white54,
                    fontSize: 10.5,
                    fontWeight: selected
                        ? FontWeight.w700
                        : FontWeight.w500,
                  ),
                ),
              ),
            ),
          );
        }).toList(),
      ),
    );
  }

  Widget _notificationCard(
    _ParentNotification notification,
  ) {
    return Material(
      color: Colors.transparent,
      child: InkWell(
        onTap: () {
          setState(() {
            notification.unread = false;
          });
        },
        borderRadius:
            BorderRadius.circular(21),
        child: Container(
          width: double.infinity,
          padding: const EdgeInsets.all(15),
          decoration: BoxDecoration(
            color: notification.unread
                ? const Color(0xFF004848)
                    .withOpacity(.78)
                : darkTeal.withOpacity(.65),
            borderRadius:
                BorderRadius.circular(21),
            border: Border.all(
              color: notification.unread
                  ? notification.color
                      .withOpacity(.20)
                  : mint.withOpacity(.08),
            ),
          ),
          child: Row(
            crossAxisAlignment:
                CrossAxisAlignment.start,
            children: [
              Container(
                width: 46,
                height: 46,
                decoration: BoxDecoration(
                  color: notification.color
                      .withOpacity(.10),
                  borderRadius:
                      BorderRadius.circular(14),
                ),
                child: Icon(
                  notification.icon,
                  color: notification.color,
                  size: 23,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Row(
                      children: [
                        Expanded(
                          child: Text(
                            notification.title,
                            maxLines: 2,
                            overflow:
                                TextOverflow
                                    .ellipsis,
                            style:
                                GoogleFonts.dmSans(
                              color: Colors.white,
                              fontSize: 12.5,
                              fontWeight:
                                  notification
                                          .unread
                                      ? FontWeight
                                          .w800
                                      : FontWeight
                                          .w600,
                            ),
                          ),
                        ),

                        if (notification.unread)
                          Container(
                            width: 8,
                            height: 8,
                            margin:
                                const EdgeInsets.only(
                              left: 8,
                            ),
                            decoration:
                                BoxDecoration(
                              color:
                                  notification.color,
                              shape: BoxShape.circle,
                            ),
                          ),
                      ],
                    ),

                    const SizedBox(height: 5),

                    Text(
                      notification.subtitle,
                      maxLines: 3,
                      overflow:
                          TextOverflow.ellipsis,
                      style: GoogleFonts.dmSans(
                        color: Colors.white.withOpacity(0.50),
                        fontSize: 10.5,
                        height: 1.4,
                      ),
                    ),

                    const SizedBox(height: 9),

                    Row(
                      children: [
                        Container(
                          padding:
                              const EdgeInsets
                                  .symmetric(
                            horizontal: 8,
                            vertical: 4,
                          ),
                          decoration:
                              BoxDecoration(
                            color:
                                notification.color
                                    .withOpacity(
                              .08,
                            ),
                            borderRadius:
                                BorderRadius
                                    .circular(20),
                          ),
                          child: Text(
                            notification.type,
                            style:
                                GoogleFonts.dmSans(
                              color:
                                  notification.color,
                              fontSize: 8.5,
                              fontWeight:
                                  FontWeight.w700,
                            ),
                          ),
                        ),

                        const SizedBox(width: 8),

                        Expanded(
                          child: Text(
                            notification.time,
                            maxLines: 1,
                            overflow:
                                TextOverflow
                                    .ellipsis,
                            textAlign:
                                TextAlign.end,
                            style:
                                GoogleFonts.dmSans(
                              color:
                                  Colors.white30,
                              fontSize: 9,
                            ),
                          ),
                        ),
                      ],
                    ),
                  ],
                ),
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildEmptyState() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.symmetric(
        horizontal: 20,
        vertical: 42,
      ),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius:
            BorderRadius.circular(22),
        border: Border.all(
          color: mint.withOpacity(.08),
        ),
      ),
      child: Column(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: teal.withOpacity(.10),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.notifications_none_rounded,
              color: mint,
              size: 27,
            ),
          ),

          const SizedBox(height: 13),

          Text(
            'No notifications',
            style: GoogleFonts.dmSans(
              color: Colors.white,
              fontSize: 14,
              fontWeight: FontWeight.w700,
            ),
          ),

          const SizedBox(height: 4),

          Text(
            'There are no notifications in this category.',
            textAlign: TextAlign.center,
            style: GoogleFonts.dmSans(
              color: Colors.white38,
              fontSize: 10.5,
            ),
          ),
        ],
      ),
    );
  }
}

class _ParentNotification {
  final String title;
  final String subtitle;
  final String time;
  final String type;
  final IconData icon;
  final Color color;
  bool unread;

  _ParentNotification({
    required this.title,
    required this.subtitle,
    required this.time,
    required this.type,
    required this.icon,
    required this.color,
    required this.unread,
  });
}
