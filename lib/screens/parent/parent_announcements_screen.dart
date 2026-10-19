import 'package:flutter/material.dart';
import 'package:google_fonts/google_fonts.dart';
import 'package:video_player/video_player.dart';

class ParentAnnouncementsScreen extends StatefulWidget {
  const ParentAnnouncementsScreen({super.key});

  @override
  State<ParentAnnouncementsScreen> createState() =>
      _ParentAnnouncementsScreenState();
}

class _ParentAnnouncementsScreenState
    extends State<ParentAnnouncementsScreen> {
  late VideoPlayerController _videoController;

  static const Color background = Color(0xFF063E40);
  static const Color darkTeal = Color(0xFF074F51);
  static const Color teal = Color(0xFF0A6B6E);
  static const Color mint = Color(0xFF5DCAA5);
  static const Color green = Color(0xFF35EE96);
  static const Color warning = Color(0xFFFFD166);
  static const Color red = Color(0xFFFF6B6B);

  String _selectedFilter = 'All';

  final List<String> _filters = const [
    'All',
    'General',
    'Fees',
    'Maintenance',
    'Important',
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
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            _buildHeaderCard(),

                            const SizedBox(height: 20),

                            _sectionTitle(
                              'Categories',
                              'Filter hostel announcements',
                            ),

                            const SizedBox(height: 12),

                            _buildFilters(),

                            const SizedBox(height: 22),

                            _sectionTitle(
                              'Latest Announcements',
                              'Updates shared by hostel administration',
                            ),

                            const SizedBox(height: 12),

                            if (_shows('General'))
                              _announcementCard(
                              icon: Icons.restaurant_menu_rounded,
                              iconColor: warning,
                              category: 'General',
                              title: 'Updated Mess Timing',
                              description:
                                  'Dinner timing has been changed to 7:30 PM - 9:30 PM from today.',
                              date: 'Today • 4:30 PM',
                              important: false,
                            ),

                            if (_shows('General'))
                              const SizedBox(height: 12),

                            if (_shows('Fees', important: true))
                              _announcementCard(
                              icon: Icons.payments_outlined,
                              iconColor: mint,
                              category: 'Fees',
                              title: 'Monthly Fee Reminder',
                              description:
                                  'Parents are requested to ensure the hostel fee is cleared before the due date.',
                              date: 'Today • 11:15 AM',
                              important: true,
                            ),

                            if (_shows('Fees', important: true))
                              const SizedBox(height: 12),

                            if (_shows('Maintenance'))
                              _announcementCard(
                              icon: Icons.build_outlined,
                              iconColor: const Color(0xFFB7A8FF),
                              category: 'Maintenance',
                              title: 'Water Supply Maintenance',
                              description:
                                  'Water supply may remain unavailable from 2:00 PM to 4:00 PM due to maintenance work.',
                              date: '25 Aug 2026 • 6:20 PM',
                              important: false,
                            ),

                            if (_shows('Maintenance'))
                              const SizedBox(height: 12),

                            if (_shows('Important', important: true))
                              _announcementCard(
                              icon: Icons.shield_outlined,
                              iconColor: red,
                              category: 'Important',
                              title: 'Updated Visitor Policy',
                              description:
                                  'All visitors must register at the reception desk before entering the hostel premises.',
                              date: '24 Aug 2026 • 2:10 PM',
                              important: true,
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
          height: _videoController.value.size.height,
          child: VideoPlayer(_videoController),
        ),
      ),
    );
  }

  Widget _buildTopBar() {
    return Padding(
      padding: const EdgeInsets.fromLTRB(
        10,
        8,
        18,
        8,
      ),
      child: Row(
        children: [
          IconButton(
            onPressed: () => Navigator.pop(context),
            icon: const Icon(
              Icons.arrow_back_ios_new_rounded,
              color: Colors.white,
              size: 20,
            ),
          ),

          const SizedBox(width: 4),

          Expanded(
            child: Text(
              'Announcements',
              overflow: TextOverflow.ellipsis,
              style: GoogleFonts.dmSans(
                color: Colors.white,
                fontSize: 18,
                fontWeight: FontWeight.w700,
              ),
            ),
          ),

          Container(
            width: 42,
            height: 42,
            decoration: BoxDecoration(
              shape: BoxShape.circle,
              color: Colors.white.withOpacity(.07),
              border: Border.all(
                color: mint.withOpacity(.14),
              ),
            ),
            child: const Icon(
              Icons.campaign_outlined,
              color: mint,
              size: 21,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildHeaderCard() {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(20),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            const Color(0xFF00645E).withOpacity(.96),
            const Color(0xFF003B3B).withOpacity(.96),
          ],
        ),
        borderRadius: BorderRadius.circular(28),
        border: Border.all(
          color: mint.withOpacity(.20),
        ),
        boxShadow: [
          BoxShadow(
            color: teal.withOpacity(.12),
            blurRadius: 24,
            spreadRadius: 1,
          ),
        ],
      ),
      child: Row(
        children: [
          Container(
            width: 58,
            height: 58,
            decoration: BoxDecoration(
              color: teal.withOpacity(.16),
              borderRadius: BorderRadius.circular(18),
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
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Hostel Updates',
                  style: GoogleFonts.dmSans(
                    color: Colors.white,
                    fontSize: 18,
                    fontWeight: FontWeight.w800,
                  ),
                ),

                const SizedBox(height: 4),

                Text(
                  'Stay informed about important hostel notices and updates.',
                  style: GoogleFonts.dmSans(
                    color: Colors.white60,
                    fontSize: 11,
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
      crossAxisAlignment: CrossAxisAlignment.start,
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
          final bool selected =
              _selectedFilter == filter;

          return Padding(
            padding: const EdgeInsets.only(right: 9),
            child: InkWell(
              onTap: () {
                setState(() {
                  _selectedFilter = filter;
                });
              },
              borderRadius: BorderRadius.circular(30),
              child: AnimatedContainer(
                duration: const Duration(
                  milliseconds: 200,
                ),
                padding: const EdgeInsets.symmetric(
                  horizontal: 15,
                  vertical: 9,
                ),
                decoration: BoxDecoration(
                  color: selected
                      ? teal.withOpacity(.20)
                      : darkTeal.withOpacity(.65),
                  borderRadius:
                      BorderRadius.circular(30),
                  border: Border.all(
                    color: selected
                        ? mint.withOpacity(.5)
                        : mint.withOpacity(.10),
                  ),
                ),
                child: Text(
                  filter,
                  style: GoogleFonts.dmSans(
                    color: selected
                        ? mint
                        : Colors.white54,
                    fontSize: 11,
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

  bool _shows(
    String category, {
    bool important = false,
  }) {
    return _selectedFilter == 'All' ||
        _selectedFilter == category ||
        (_selectedFilter == 'Important' && important);
  }

  Widget _announcementCard({
    required IconData icon,
    required Color iconColor,
    required String category,
    required String title,
    required String description,
    required String date,
    required bool important,
  }) {
    return Container(
      width: double.infinity,
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.black.withOpacity(.32),
        borderRadius: BorderRadius.circular(22),
        border: Border.all(
          color: important
              ? warning.withOpacity(.24)
              : mint.withOpacity(.10),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            crossAxisAlignment:
                CrossAxisAlignment.start,
            children: [
              Container(
                width: 44,
                height: 44,
                decoration: BoxDecoration(
                  color:
                      iconColor.withOpacity(.10),
                  borderRadius:
                      BorderRadius.circular(14),
                ),
                child: Icon(
                  icon,
                  color: iconColor,
                  size: 22,
                ),
              ),

              const SizedBox(width: 12),

              Expanded(
                child: Column(
                  crossAxisAlignment:
                      CrossAxisAlignment.start,
                  children: [
                    Wrap(
                      spacing: 7,
                      runSpacing: 6,
                      children: [
                        _categoryChip(
                          category,
                          iconColor,
                        ),
                        if (important)
                          _categoryChip(
                            'Important',
                            warning,
                          ),
                      ],
                    ),

                    const SizedBox(height: 8),

                    Text(
                      title,
                      style: GoogleFonts.dmSans(
                        color: Colors.white,
                        fontSize: 14,
                        fontWeight:
                            FontWeight.w800,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),

          const SizedBox(height: 12),

          Text(
            description,
            style: GoogleFonts.dmSans(
              color: Colors.white60,
              fontSize: 11.5,
              height: 1.5,
            ),
          ),

          const SizedBox(height: 14),

          Divider(
            height: 1,
            color: Colors.white.withOpacity(.06),
          ),

          const SizedBox(height: 11),

          Row(
            children: [
              const Icon(
                Icons.schedule_rounded,
                color: Colors.white30,
                size: 16,
              ),

              const SizedBox(width: 6),

              Expanded(
                child: Text(
                  date,
                  maxLines: 1,
                  overflow:
                      TextOverflow.ellipsis,
                  style: GoogleFonts.dmSans(
                    color: Colors.white38,
                    fontSize: 10,
                  ),
                ),
              ),

              Text(
                'Read More',
                style: GoogleFonts.dmSans(
                  color: mint,
                  fontSize: 10.5,
                  fontWeight: FontWeight.w700,
                ),
              ),

              const SizedBox(width: 3),

              const Icon(
                Icons.chevron_right_rounded,
                color: mint,
                size: 18,
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _categoryChip(
    String text,
    Color color,
  ) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: 9,
        vertical: 5,
      ),
      decoration: BoxDecoration(
        color: color.withOpacity(.10),
        borderRadius: BorderRadius.circular(20),
      ),
      child: Text(
        text,
        style: GoogleFonts.dmSans(
          color: color,
          fontSize: 9,
          fontWeight: FontWeight.w700,
        ),
      ),
    );
  }
}
